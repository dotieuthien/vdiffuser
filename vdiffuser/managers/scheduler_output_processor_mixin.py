from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from vdiffuser.managers.scheduler import (
        DiffusionBatchResult,
        TextEncodingBatchResult,
        VAEEncodingBatchResult,
        VAEDecodingBatchResult,
        ScheduleBatch,
        Scheduler,
    )
    from vdiffuser.managers.io_struct import AbortReq, BatchImageOut
    from vdiffuser.managers.schedule_batch import BaseFinishReason, DiffusionReq

logger = logging.getLogger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 5  # For image generation, smaller interval


class DiffusionSchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Diffusion Scheduler.
    Adapted from SGLang for image diffusion pipeline:
    text encode -> image encode (VAE) -> diffusion denoise -> image decode (VAE)
    """

    def process_batch_result_text_encoding(
        self: Scheduler,
        batch: ScheduleBatch,
        result: TextEncodingBatchResult,
        launch_done: Optional[threading.Event] = None,
    ):
        """Process text encoding results (first stage of diffusion pipeline)."""
        text_embeddings = result.text_embeddings

        if self.enable_overlap:
            text_embeddings = self.tp_worker.resolve_last_batch_result(
                launch_done)
        else:
            # Move embeddings to CPU if needed
            if isinstance(text_embeddings, torch.Tensor):
                text_embeddings = text_embeddings.cpu()

        # Update requests with text embeddings
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            req.text_embeddings = text_embeddings[i]
            req.pipeline_stage = "text_encoded"

            # Mark as ready for next stage if not chunked
            if req.is_chunked <= 0:
                req.ready_for_vae_encoding = True
            else:
                req.is_chunked -= 1

        # Stream intermediate results if requested
        self.stream_output_text_encoding(batch.reqs)

    def process_batch_result_vae_encoding(
        self: Scheduler,
        batch: ScheduleBatch,
        result: VAEEncodingBatchResult,
        launch_done: Optional[threading.Event] = None,
    ):
        """Process VAE encoding results (second stage - image to latent)."""
        latent_embeddings = result.latent_embeddings

        if self.enable_overlap:
            latent_embeddings = self.tp_worker.resolve_last_batch_result(
                launch_done)
        else:
            if isinstance(latent_embeddings, torch.Tensor):
                latent_embeddings = latent_embeddings.cpu()

        # Update requests with latent embeddings
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            req.latent_embeddings = latent_embeddings[i]
            req.pipeline_stage = "vae_encoded"

            if req.is_chunked <= 0:
                req.ready_for_diffusion = True
            else:
                req.is_chunked -= 1

        self.stream_output_vae_encoding(batch.reqs)

    def process_batch_result_diffusion_denoise(
        self: Scheduler,
        batch: ScheduleBatch,
        result: DiffusionBatchResult,
        launch_done: Optional[threading.Event] = None,
    ):
        """Process diffusion denoising results (third stage - main diffusion process)."""
        denoised_latents, timestep_completed, can_run_cuda_graph = (
            result.denoised_latents,
            result.timestep_completed,
            result.can_run_cuda_graph,
        )

        self.num_diffusion_steps += len(batch.reqs)

        if self.enable_overlap:
            denoised_latents, timestep_completed, can_run_cuda_graph = (
                self.tp_worker.resolve_last_batch_result(launch_done)
            )
        else:
            if isinstance(denoised_latents, torch.Tensor):
                denoised_latents = denoised_latents.cpu()

        self.memory_pool.free_group_begin()

        # Check completion condition for diffusion steps
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            if self.enable_overlap and req.finished_diffusion():
                # Free memory for completed diffusion
                if hasattr(batch, 'latent_cache_loc'):
                    self.memory_pool.free(batch.latent_cache_loc[i:i+1])
                continue

            # Update denoised latents
            req.denoised_latents = denoised_latents[i]
            req.current_timestep = timestep_completed
            req.diffusion_steps_completed += 1

            # Check if diffusion is complete
            if req.diffusion_steps_completed >= req.total_diffusion_steps:
                req.pipeline_stage = "diffusion_complete"
                req.ready_for_vae_decoding = True
                req.time_stats.diffusion_completion_time = time.time()

            req.check_diffusion_finished()

        self.stream_output_diffusion(batch.reqs)
        self.memory_pool.free_group_end()

        self.forward_ct_diffusion = (self.forward_ct_diffusion + 1) % (1 << 30)
        if (
            self.current_scheduler_metrics_enabled()
            and self.forward_ct_diffusion % self.server_args.diffusion_log_interval == 0
        ):
            self.log_diffusion_stats(can_run_cuda_graph, running_batch=batch)

    def process_batch_result_vae_decoding(
        self: Scheduler,
        batch: ScheduleBatch,
        result: VAEDecodingBatchResult,
        launch_done: Optional[threading.Event] = None,
    ):
        """Process VAE decoding results (final stage - latent to image)."""
        decoded_images = result.decoded_images

        if self.enable_overlap:
            decoded_images = self.tp_worker.resolve_last_batch_result(
                launch_done)
        else:
            # Convert to PIL Images or numpy arrays
            if isinstance(decoded_images, torch.Tensor):
                decoded_images = self._tensor_to_images(decoded_images)

        # Finalize requests with generated images
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            req.generated_image = decoded_images[i]
            req.pipeline_stage = "complete"
            req.check_finished()

            if req.finished():
                req.time_stats.completion_time = time.time()
                # Cache completed request if needed
                if hasattr(self, 'image_cache'):
                    self.image_cache.cache_finished_req(req)

        self.stream_output_final_images(batch.reqs)

    def _tensor_to_images(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert tensor to PIL Images."""
        # Assuming tensor is in shape [batch, channels, height, width]
        # and values are in range [-1, 1] or [0, 1]
        images = []

        for i in range(tensor.shape[0]):
            img_tensor = tensor[i]

            # Normalize to [0, 1] if in [-1, 1]
            if img_tensor.min() < 0:
                img_tensor = (img_tensor + 1) / 2

            # Convert to numpy and scale to [0, 255]
            img_np = (img_tensor.permute(1, 2, 0).numpy()
                      * 255).astype(np.uint8)

            # Convert to PIL Image
            img_pil = Image.fromarray(img_np)
            images.append(img_pil)

        return images

    def stream_output_text_encoding(self, reqs: List[DiffusionReq]):
        """Stream text encoding outputs."""
        if not self.stream_intermediate_results:
            return

        rids = []
        text_embeddings = []

        for req in reqs:
            if req.text_embeddings is not None and req.stream:
                rids.append(req.rid)
                text_embeddings.append(req.text_embeddings.tolist() if isinstance(
                    req.text_embeddings, torch.Tensor) else req.text_embeddings)

        if rids:
            self.send_to_client.send_pyobj({
                'type': 'text_encoding_complete',
                'rids': rids,
                'text_embeddings': text_embeddings,
            })

    def stream_output_vae_encoding(self, reqs: List[DiffusionReq]):
        """Stream VAE encoding outputs."""
        if not self.stream_intermediate_results:
            return

        rids = []
        latent_shapes = []

        for req in reqs:
            if req.latent_embeddings is not None and req.stream:
                rids.append(req.rid)
                latent_shapes.append(list(req.latent_embeddings.shape) if hasattr(
                    req.latent_embeddings, 'shape') else None)

        if rids:
            self.send_to_client.send_pyobj({
                'type': 'vae_encoding_complete',
                'rids': rids,
                'latent_shapes': latent_shapes,
            })

    def stream_output_diffusion(self, reqs: List[DiffusionReq]):
        """Stream diffusion progress outputs."""
        rids = []
        diffusion_progress = []
        intermediate_images = []

        for req in reqs:
            should_output = False

            if req.finished_diffusion():
                should_output = True
            elif req.stream:
                # Stream at specified intervals during diffusion
                stream_interval = req.diffusion_params.stream_interval or self.diffusion_stream_interval
                should_output = req.diffusion_steps_completed % stream_interval == 0

            if should_output:
                rids.append(req.rid)
                progress = req.diffusion_steps_completed / req.total_diffusion_steps
                diffusion_progress.append(progress)

                # Optionally include intermediate images
                if req.return_intermediate_images and hasattr(req, 'denoised_latents'):
                    # Quick decode for preview (lower quality)
                    intermediate_img = self._quick_decode_latent(
                        req.denoised_latents)
                    intermediate_images.append(intermediate_img)
                else:
                    intermediate_images.append(None)

        if rids:
            self.send_to_client.send_pyobj({
                'type': 'diffusion_progress',
                'rids': rids,
                'progress': diffusion_progress,
                'intermediate_images': intermediate_images,
            })

    def stream_output_final_images(self, reqs: List[DiffusionReq]):
        """Stream final generated images."""
        rids = []
        finished_reasons: List[BaseFinishReason] = []
        generated_images = []
        prompt_tokens = []
        diffusion_steps = []
        seed_used = []

        for req in reqs:
            if req.finished() and req.generated_image is not None:
                if req.finished_output:
                    # Prevent duplicate outputs
                    continue
                req.finished_output = True

                rids.append(req.rid)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )

                # Convert PIL Image to base64 or save path
                if isinstance(req.generated_image, Image.Image):
                    # Convert to bytes or save and provide path
                    generated_images.append(
                        self._image_to_output_format(req.generated_image))
                else:
                    generated_images.append(req.generated_image)

                prompt_tokens.append(len(req.prompt_tokens)
                                     if hasattr(req, 'prompt_tokens') else 0)
                diffusion_steps.append(req.diffusion_steps_completed)
                seed_used.append(req.seed)

            if (
                req.finished()
                and self.tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        # Send to client
        if rids:
            self.send_to_client.send_pyobj(
                BatchImageOut(
                    rids,
                    finished_reasons,
                    generated_images,
                    prompt_tokens,
                    diffusion_steps,
                    seed_used,
                )
            )

    def _quick_decode_latent(self, latent: torch.Tensor) -> Optional[Image.Image]:
        """Quick decode latent for intermediate preview (lower quality)."""
        try:
            # Use a simplified/cached VAE decoder for preview
            if hasattr(self, 'preview_vae_decoder'):
                with torch.no_grad():
                    preview_img = self.preview_vae_decoder(latent.unsqueeze(0))
                    return self._tensor_to_images(preview_img)[0]
        except Exception as e:
            logger.warning(f"Failed to generate preview image: {e}")
        return None

    def _image_to_output_format(self, image: Image.Image) -> str:
        """Convert PIL Image to output format (base64, file path, etc.)."""
        import io
        import base64

        # Convert to base64 for JSON serialization
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_data = buffer.getvalue()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return img_base64

    def stream_output(
        self: Scheduler,
        reqs: List[DiffusionReq],
        stage: str,
        skip_req: Optional[DiffusionReq] = None,
    ):
        """Main stream output dispatcher based on pipeline stage."""
        if stage == "text_encoding":
            self.stream_output_text_encoding(
                [req for req in reqs if req != skip_req])
        elif stage == "vae_encoding":
            self.stream_output_vae_encoding(
                [req for req in reqs if req != skip_req])
        elif stage == "diffusion":
            self.stream_output_diffusion(
                [req for req in reqs if req != skip_req])
        elif stage == "vae_decoding" or stage == "complete":
            self.stream_output_final_images(
                [req for req in reqs if req != skip_req])
        else:
            logger.warning(f"Unknown pipeline stage for streaming: {stage}")
