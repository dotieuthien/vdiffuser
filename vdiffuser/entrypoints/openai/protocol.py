from __future__ import annotations
import time
import os
from typing import Union, Optional, TYPE_CHECKING, Tuple, Mapping, IO, List, Dict, Literal, Any
from pydantic import BaseModel, Field, model_serializer

if TYPE_CHECKING:
    Base64FileInput = Union[IO[bytes], os.PathLike[str]]
    FileContent = Union[IO[bytes], bytes, os.PathLike[str]]
else:
    Base64FileInput = Union[IO[bytes], os.PathLike]
    FileContent = Union[IO[bytes], bytes, os.PathLike]

FileTypes = Union[
    FileContent,
    Tuple[Optional[str], FileContent],
    Tuple[Optional[str], FileContent, Optional[str]],
    Tuple[Optional[str], FileContent, Optional[str], Mapping[str, str]],
]

ImageModel = Literal["dall-e-2", "dall-e-3", "gpt-image-1"]

class ImageGenerateParamsBase(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    prompt: str
    background: Optional[Literal["transparent", "opaque", "auto"]] = None
    model: Optional[Union[str, ImageModel]] = None
    moderation: Optional[Literal["low", "auto"]] = None
    n: Optional[int] = Field(None, ge=1)
    output_compression: Optional[int] = Field(None, ge=0, le=100)
    output_format: Optional[Literal["png", "jpeg", "webp"]] = None
    partial_images: Optional[int] = Field(None, ge=0)
    quality: Optional[Literal["standard", "hd", "low", "medium", "high", "auto"]] = None
    response_format: Optional[Literal["url", "b64_json"]] = None
    size: Optional[
        Literal["auto", "1024x1024", "1536x1024", "1024x1536", "256x256", "512x512", "1792x1024", "1024x1792"]
    ] = None
    style: Optional[Literal["vivid", "natural"]] = None
    user: Optional[str] = None

class ImageGenerateParamsNonStreaming(ImageGenerateParamsBase):
    stream: Optional[Literal[False]] = False

class ImageGenerateParamsStreaming(ImageGenerateParamsBase):
    stream: Literal[True] = True

ImageGenerateParams = Union[ImageGenerateParamsNonStreaming, ImageGenerateParamsStreaming]

class ImageEditParamsBase(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    
    image: Union[FileTypes, List[FileTypes]]
    prompt: str
    background: Optional[Literal["transparent", "opaque", "auto"]] = None
    input_fidelity: Optional[Literal["high", "low"]] = None
    mask: Optional[FileTypes] = None
    model: Optional[Union[str, ImageModel]] = None
    n: Optional[int] = Field(None, ge=1)
    output_compression: Optional[int] = Field(None, ge=0, le=100)
    output_format: Optional[Literal["png", "jpeg", "webp"]] = None
    partial_images: Optional[int] = Field(None, ge=0)
    quality: Optional[Literal["standard", "low", "medium", "high", "auto"]] = None
    response_format: Optional[Literal["url", "b64_json"]] = None
    size: Optional[Literal["256x256", "512x512", "1024x1024", "1536x1024", "1024x1536", "auto"]] = None
    user: Optional[str] = None

    # @model_validator(mode='after')
    # def validate_image_input(cls, values):
    #     if not values.image:
    #         raise ValueError("At least one image is required")
    #     return values

class ImageEditParamsNonStreaming(ImageEditParamsBase):
    stream: Optional[Literal[False]] = False

class ImageEditParamsStreaming(ImageEditParamsBase):
    stream: Literal[True] = True

ImageEditParams = Union[ImageEditParamsNonStreaming, ImageEditParamsStreaming]

OpenAIServingRequest = Union[ImageEditParams, ImageGenerateParams]


class LogProbs(BaseModel):
    tokens: List[str] = Field(default_factory=list)
    token_logprobs: List[float] = Field(default_factory=list)
    text_offset: List[int] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: Optional[str] = None
    max_model_len: Optional[int] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)

class ImageUsageInfo(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int
    input_tokens_details: Optional[Dict[str, int]] = None

class ImageResponseData(BaseModel):
    b64_json: str

class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageResponseData]
    usage: ImageUsageInfo

class BaseImageEvent(BaseModel):
    type: str

class ImagePartialEvent(BaseImageEvent):
    b64_json: str
    partial_image_index: int

class ImageCompletionEvent(BaseImageEvent):
    b64_json: str
    usage: Optional[ImageUsageInfo] = None

class ImageGenerationPartialEvent(ImagePartialEvent):
    type: Literal["image_generation.partial_image"] = "image_generation.partial_image"

class ImageGenerationCompletedEvent(ImageCompletionEvent):
    type: Literal["image_generation.completed"] = "image_generation.completed"

class ImageEditPartialEvent(ImagePartialEvent):
    type: Literal["image_edit.partial_image"] = "image_edit.partial_image"

class ImageEditCompletedEvent(ImageCompletionEvent):
    type: Literal["image_edit.completed"] = "image_edit.completed"