import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from vdiffuser.entrypoints.openai.protocol import (
    ErrorResponse,
    ImageGenerateRequest,
    ImageGenerationCompletedEvent,
    ImageGenerationPartialEvent,
    ImageGenerationResponse,
    ImageResponseData,
    ImageUsageInfo,
)
from vdiffuser.entrypoints.openai.serving_base import OpenAIServingBase
from vdiffuser.managers.io_struct import GenerateReqInput
from vdiffuser.managers.template_manager import TemplateManager

logger = logging.getLogger(__name__)

# A base64 encoded 1x1 red pixel PNG
DUMMY_B64_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB/epv2AAAAABJRU5ErkJggg=="


class OpenAIServingImagesGenerate(OpenAIServingBase):
    """Handler for /v1/images/generations requests"""

    def __init__(
        self,
        template_manager: TemplateManager,
    ):
        self.template_manager = template_manager

    def _request_id_prefix(self) -> str:
        return "image_generate-"

    async def _convert_to_internal_request(
        self,
        request: ImageGenerateRequest,
    ) -> tuple[GenerateReqInput, ImageGenerateRequest]:
        """Convert OpenAI image generate request to internal format"""

        sampling_params = self._build_sampling_params(request)

        prompt_kwargs = {"text": request.prompt}

        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            sampling_params=sampling_params,
            stream=request.stream,
            rid=getattr(request, "rid", None),
        )

        return adapted_request, request

    def _build_sampling_params(self, request: ImageGenerateRequest) -> Dict[str, Any]:
        """Build sampling parameters for the request"""
        sampling_params = {
            "n": request.n,
            "quality": request.quality,
            "size": request.size,
            "style": request.style,
            "user": request.user,
            "response_format": request.response_format,
            "output_format": request.output_format,
            "output_compression": request.output_compression,
            "partial_images": request.partial_images,
            "background": request.background,
            "moderation": request.moderation,
            "model": request.model,
        }
        return {k: v for k, v in sampling_params.items() if v is not None}

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ImageGenerateRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming image generation request"""
        return StreamingResponse(
            self._generate_image_stream(adapted_request, request, raw_request),
            media_type="text/event-stream",
        )

    async def _generate_image_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ImageGenerateRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming image generation response"""
        
        # Yield a few partial events
        for i in range(3):
            partial_event = ImageGenerationPartialEvent(
                b64_json=DUMMY_B64_IMAGE,
                partial_image_index=i,
            )
            yield f"data: {partial_event.model_dump_json()}\n\n"
            await asyncio.sleep(0.1)

        # Yield the final completion event
        usage = ImageUsageInfo(total_tokens=10, input_tokens=5, output_tokens=5)
        completed_event = ImageGenerationCompletedEvent(
            b64_json=DUMMY_B64_IMAGE,
            usage=usage,
        )
        yield f"data: {completed_event.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ImageGenerateRequest,
        raw_request: Request,
    ) -> Union[ImageGenerationResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming image generation request"""
        
        await asyncio.sleep(0.1) # Simulate work
        
        n_images = request.n or 1
        
        data = [ImageResponseData(b64_json=DUMMY_B64_IMAGE) for _ in range(n_images)]
        usage = ImageUsageInfo(total_tokens=10 * n_images, input_tokens=5 * n_images, output_tokens=5 * n_images)

        return ImageGenerationResponse(
            created=int(time.time()),
            data=data,
            usage=usage,
        )