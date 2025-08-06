from __future__ import annotations
from typing import Union, Optional, TYPE_CHECKING, Tuple, Mapping, IO, PathLike
from typing_extensions import Literal, Required, TypedDict
if TYPE_CHECKING:
    Base64FileInput = Union[IO[bytes], PathLike[str]]
    FileContent = Union[IO[bytes], bytes, PathLike[str]]
else:
    Base64FileInput = Union[IO[bytes], PathLike]
    FileContent = Union[IO[bytes], bytes, PathLike]
FileTypes = Union[
    FileContent,
    Tuple[Optional[str], FileContent],
    Tuple[Optional[str], FileContent, Optional[str]],
    Tuple[Optional[str], FileContent, Optional[str], Mapping[str, str]],
]

from typing_extensions import Literal, TypeAlias

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

from __future__ import annotations
from typing import List, Union, Optional
from typing_extensions import Literal, Required, TypedDict

from .._types import FileTypes
from .image_model import ImageModel

__all__ = ["ImageEditParamsBase", "ImageEditParamsNonStreaming", "ImageEditParamsStreaming"]


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