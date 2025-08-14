from __future__ import annotations

import logging
import time
import torch
from fastapi import FastAPI, APIRouter
from diffusers import StableDiffusionXLPipeline
from schemas import GenerateRequest, GenerateResponse
from utils import benchmark_fn, flush

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CKPT_ID = "stabilityai/stable-diffusion-xl-base-1.0"
NUM_WARMUP_RUNS = 3

pipeline = None

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

@router.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}

@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):    
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized. Please restart the server.")
    
    flush()
    
    def generate_image():
        return pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
        ).images[0]
    
    time_s = benchmark_fn(generate_image)
    
    mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    mem_gb = round(mem_gb, 2)
    
    return GenerateResponse(
        prompt=req.prompt,
        time_s=time_s,
        mem_gb=mem_gb,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        width=req.width,
        height=req.height,
    )

def load_and_warmup_pipeline():
    global pipeline
    
    logger.info("Loading pipeline: %s", CKPT_ID)
    start_time = time.time()
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        CKPT_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipeline = pipeline.to("cuda")
    
    load_time = time.time() - start_time
    logger.info("Pipeline loaded in %.2fs", load_time)
    
    logger.info("Starting warmup with %d runs...", NUM_WARMUP_RUNS)
    warmup_prompt = "a simple test image"
    
    warmup_start = time.time()
    for i in range(NUM_WARMUP_RUNS):
        logger.info("Warmup run %d/%d", i+1, NUM_WARMUP_RUNS)
        with torch.no_grad():
            _ = pipeline(
                prompt=warmup_prompt,
                num_inference_steps=10,
                width=512,
                height=512,
            ).images[0]
        flush()
    
    warmup_time = time.time() - warmup_start
    logger.info("Warmup completed in %.2fs", warmup_time)
    logger.info("Pipeline ready for benchmarking!")

app = FastAPI(
    title="Diffusers FastAPI Benchmark", 
    description="FastAPI benchmark server for diffusion models",
    openapi_url="/fastapi/openapi.json"
)

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    load_and_warmup_pipeline()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)