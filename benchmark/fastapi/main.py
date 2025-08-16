import asyncio
import logging
import os
import random
import tempfile
import traceback

import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from diffusers import StableDiffusionXLPipeline
from schemas import GenerateRequest, GenerateResponse
from utils import benchmark_fn, flush, calculate_params

logger = logging.getLogger(__name__)

CKPT_ID = "stabilityai/stable-diffusion-xl-base-1.0"

class TextToImagePipeline:
    pipeline: StableDiffusionXLPipeline = None
    device: str = None

    def start(self):
        logger.info("Loading CUDA")
        self.device = "cuda"
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            CKPT_ID,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)


app = FastAPI()

shared_pipeline = TextToImagePipeline()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    shared_pipeline.start()

@app.post("/benchmark/generate")
async def benchmark_generate(request: GenerateRequest) -> GenerateResponse:
    try:
        flush()
        
        torch.cuda.reset_peak_memory_stats()
        start_memory = psutil.Process().memory_info().rss / (1024**3)  # GB
        
        loop = asyncio.get_event_loop()
        
        scheduler = shared_pipeline.pipeline.scheduler.from_config(
            shared_pipeline.pipeline.scheduler.config
        )
        pipeline = StableDiffusionXLPipeline.from_pipe(
            shared_pipeline.pipeline, scheduler=scheduler
        )
        
        model_params = calculate_params(pipeline.unet) if hasattr(pipeline, 'unet') else None
        
        generator = torch.Generator(device=shared_pipeline.device)
        generator.manual_seed(random.randint(0, 10000000))
        
        def _generate():
            return pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                generator=generator
            )
        
        generation_time = await loop.run_in_executor(
            None, 
            lambda: benchmark_fn(_generate)
        )
        
        output = await loop.run_in_executor(None, _generate)
        
        end_memory = psutil.Process().memory_info().rss / (1024**3)  # GB
        peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        memory_used = max(peak_gpu_memory, end_memory - start_memory)
        
        logger.info(
            f"Generated image in {generation_time:.3f}s, "
            f"memory: {memory_used:.2f}GB, "
            f"model params: {f'{model_params:,}' if model_params is not None else 'N/A'}"
        )
        
        return GenerateResponse(
            prompt=request.prompt,
            time_s=generation_time,
            mem_gb=memory_used,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            model_params=model_params
        )
        
    except Exception as e:
        logger.error(f"Generation failed after {generation_time:.2f}s: {str(e)}")
        logger.error(traceback.format_exc())
        
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, 
            detail=f"Generation failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)