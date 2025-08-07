from __future__ import annotations
import time
import os
from typing import Union, Optional, TYPE_CHECKING, Tuple, Mapping, IO, List, Dict, Literal, Required, TypedDict
from pydantic import BaseModel, Field
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

from typing import Literal, TypeAlias

__all__ = ["ImageModel"]

ImageModel: TypeAlias = Literal["dall-e-2", "dall-e-3", "gpt-image-1"]
__all__ = ["ImageGenerateParamsBase", "ImageGenerateParamsNonStreaming", "ImageGenerateParamsStreaming"]


class ImageGenerateParamsBase(TypedDict, total=False):
    prompt: Required[str]
    background: Optional[Literal["transparent", "opaque", "auto"]]
    model: Union[str, ImageModel, None]
    moderation: Optional[Literal["low", "auto"]]
    n: Optional[int]
    output_compression: Optional[int]
    output_format: Optional[Literal["png", "jpeg", "webp"]]
    partial_images: Optional[int]
    quality: Optional[Literal["standard", "hd", "low", "medium", "high", "auto"]]
    response_format: Optional[Literal["url", "b64_json"]]
    size: Optional[
        Literal["auto", "1024x1024", "1536x1024", "1024x1536", "256x256", "512x512", "1792x1024", "1024x1792"]
    ]
    style: Optional[Literal["vivid", "natural"]]
    user: str


class ImageGenerateParamsNonStreaming(ImageGenerateParamsBase, total=False):
    stream: Optional[Literal[False]]


class ImageGenerateParamsStreaming(ImageGenerateParamsBase):
    stream: Required[Literal[True]]


ImageGenerateParams = Union[ImageGenerateParamsNonStreaming, ImageGenerateParamsStreaming]

# Image edit parameters
class ImageEditParamsBase(TypedDict, total=False):
    image: Required[Union[FileTypes, List[FileTypes]]]
    prompt: Required[str]
    background: Optional[Literal["transparent", "opaque", "auto"]]
    input_fidelity: Optional[Literal["high", "low"]]
    mask: FileTypes
    model: Union[str, ImageModel, None]
    n: Optional[int]
    output_compression: Optional[int]
    output_format: Optional[Literal["png", "jpeg", "webp"]]
    partial_images: Optional[int]
    quality: Optional[Literal["standard", "low", "medium", "high", "auto"]]
    response_format: Optional[Literal["url", "b64_json"]]
    size: Optional[Literal["256x256", "512x512", "1024x1024", "1536x1024", "1024x1536", "auto"]]
    user: str


class ImageEditParamsNonStreaming(ImageEditParamsBase, total=False):
    stream: Optional[Literal[False]]


class ImageEditParamsStreaming(ImageEditParamsBase):
    stream: Required[Literal[True]]


ImageEditParams = Union[ImageEditParamsNonStreaming, ImageEditParamsStreaming]

OpenAIServingRequest = Union[ImageEditParams, ImageGenerateParams]

# Add missing classes for compatibility with utils.py
class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    return_hidden_states: Optional[bool] = False


class CompletionRequest(BaseModel):
    """Completion request model."""
    return_hidden_states: Optional[bool] = False


class LogProbs(BaseModel):
    """Log probabilities model."""
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
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: Optional[str] = None
    max_model_len: Optional[int] = None


class ModelList(BaseModel):
    """Model list consists of model cards."""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)