import copy
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.conversation import generate_chat_conv
# from sglang.srt.entrypoints.openai.protocol import (
#     ChatCompletionRequest,
#     ChatCompletionResponse,
#     ChatCompletionResponseChoice,
#     ChatCompletionResponseStreamChoice,
#     ChatCompletionStreamResponse,
#     ChatCompletionTokenLogprob,
#     ChatMessage,
#     ChoiceLogprobs,
#     DeltaMessage,
#     ErrorResponse,
#     FunctionResponse,
#     LogProbs,
#     MessageProcessingResult,
#     ToolCall,
#     TopLogprob,
# )
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    process_hidden_states_from_ret,
    to_openai_style_logprobs,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.jinja_template_utils import process_content_for_template_format
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.utils import convert_json_schema_to_str

logger = logging.getLogger(__name__)


class OpenAIServingImagesEdit(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    def __init__(
        self, tokenizer_manager: TokenizerManager, template_manager: TemplateManager
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager

    