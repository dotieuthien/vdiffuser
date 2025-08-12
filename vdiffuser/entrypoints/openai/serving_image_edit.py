import asyncio
import base64
import io
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from vdiffuser.entrypoints.openai.protocol import (
    ErrorResponse,
    FileTypes,
    ImageEditCompletedEvent,
    ImageEditRequest,
    ImageEditPartialEvent,
    ImageGenerationResponse,
    ImageResponseData,
    ImageUsageInfo,
)
from vdiffuser.entrypoints.openai.serving_base import OpenAIServingBase
from vdiffuser.managers.io_struct import GenerateReqInput
from vdiffuser.managers.pipeline_manager import PipelineManager


logger = logging.getLogger(__name__)

# A base64 encoded 1x1 red pixel PNG
DUMMY_B64_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="


class OpenAIServingImagesEdit(OpenAIServingBase):
    """Handler for /v1/images/edits requests"""

    def __init__(
        self,
        pipeline_manager: PipelineManager
    ):
        self.pipeline_manager = pipeline_manager

    def _request_id_prefix(self) -> str:
        return "image_edit_"

    async def _read_file_to_base64(self, file_input: FileTypes) -> str:
        """Reads various file input types and returns a base64 string."""
        content = None
        if isinstance(file_input, tuple):
            file_content = file_input[1]
        else:
            file_content = file_input

        if isinstance(file_content, bytes):
            content = file_content
        elif isinstance(file_content, io.IOBase):
            content = file_content.read()
        elif isinstance(file_content, os.PathLike):
            with open(file_content, "rb") as f:
                content = f.read()
        elif hasattr(file_content, "read"):
            content = file_content.read()
        else:
            raise TypeError(f"Unsupported file input type: {type(file_content)}")

        return base64.b64encode(content).decode("utf-8")

    async def _convert_to_internal_request(
        self,
        request: ImageEditRequest,
    ) -> tuple[GenerateReqInput, ImageEditRequest]:
        """Convert OpenAI image edit request to internal format"""

        image_b64_list = []
        if isinstance(request.image, list):
            for img in request.image:
                image_b64_list.append(await self._read_file_to_base64(img))
        else:
            image_b64_list.append(await self._read_file_to_base64(request.image))

        mask_b64 = None
        if request.mask:
            mask_b64 = await self._read_file_to_base64(request.mask)

        sampling_params = self._build_sampling_params(request)

        image_data = {"images": image_b64_list}
        if mask_b64:
            image_data["mask"] = mask_b64

        adapted_request = GenerateReqInput(
            text=request.prompt,
            image_data=image_data,
            sampling_params=sampling_params,
            stream=request.stream,
            rid=getattr(request, "rid", None),
        )

        return adapted_request, request

    def _build_sampling_params(self, request: ImageEditRequest) -> Dict[str, Any]:
        """Build sampling parameters for the request"""
        sampling_params = {
            "n": request.n,
            "size": request.size,
            "user": request.user,
            "response_format": request.response_format,
            "output_format": request.output_format,
            "output_compression": request.output_compression,
            "partial_images": request.partial_images,
            "background": request.background,
            "input_fidelity": request.input_fidelity,
            "quality": request.quality,
            "model": request.model,
        }
        return {k: v for k, v in sampling_params.items() if v is not None}

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ImageEditRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming image edit request"""
        return StreamingResponse(
            self._generate_image_stream(adapted_request, request, raw_request),
            media_type="text/event-stream",
        )

    async def _generate_image_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ImageEditRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming image edit response"""
        
        # Yield a few partial events
        for i in range(3):
            partial_event = ImageEditPartialEvent(
                b64_json=DUMMY_B64_IMAGE,
                partial_image_index=i,
            )
            yield f"data: {partial_event.model_dump_json()}\n\n"
            await asyncio.sleep(0.1)

        # Yield the final completion event
        usage = ImageUsageInfo(total_tokens=15, input_tokens=10, output_tokens=5)
        completed_event = ImageEditCompletedEvent(
            b64_json=DUMMY_B64_IMAGE,
            usage=usage,
        )
        yield f"data: {completed_event.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ImageEditRequest,
        raw_request: Request,
    ) -> Union[ImageGenerationResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming image edit request"""
        
        await asyncio.sleep(0.1) # Simulate work
        
        n_images = request.n or 1
        
        data = [ImageResponseData(b64_json=DUMMY_B64_IMAGE) for _ in range(n_images)]
        usage = ImageUsageInfo(total_tokens=15 * n_images, input_tokens=10 * n_images, output_tokens=5 * n_images)

        return ImageGenerationResponse(
            created=int(time.time()),
            data=data,
            usage=usage,
        )
