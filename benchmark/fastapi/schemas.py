from pydantic import BaseModel
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024

class GenerateResponse(BaseModel):
    prompt: str
    time_s: float
    mem_gb: float
    num_inference_steps: int
    guidance_scale: float
    width: int
    height: int
    model_params: Optional[int] = None