### FastAPI benchmark for SDXL (Locust)

This folder contains a minimal FastAPI server that exposes a single endpoint to generate an image using Stable Diffusion XL, plus a Locust workload to benchmark end-to-end latency.

### Setup
From the project root or this directory, create a virtual environment and install dependencies:

```bash
cd benchmark/fastapi
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If you need a specific CUDA wheel for PyTorch, install it before `-r requirements.txt` or pin it afterwards.
- Model ID used: `stabilityai/stable-diffusion-xl-base-1.0` (defined in `main.py`).

### Start the FastAPI server
Run one of the following in this directory:

```bash
# Simple Python entrypoint
python main.py
```

or using Uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

On startup the server will initialize the SDXL pipeline on GPU. The default endpoint is available at `http://localhost:8000`.

### API
- **POST** `/benchmark/generate`
  - Request body (see `schemas.py`):

    ```json
    {
      "prompt": "a beautiful landscape with mountains and lakes at sunset",
      "negative_prompt": null,
      "num_inference_steps": 20,
      "guidance_scale": 7.5,
      "width": 1024,
      "height": 1024
    }
    ```

  - Response body:

    ```json
    {
      "prompt": "...",
      "time_s": 1.234,
      "mem_gb": 7.89,
      "num_inference_steps": 20,
      "guidance_scale": 7.5,
      "width": 1024,
      "height": 1024,
      "model_params": 123456789
    }
    ```

Example request with `curl`:

```bash
curl -X POST http://localhost:8000/benchmark/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute cat sitting on a table",
    "num_inference_steps": 10,
    "width": 512,
    "height": 512,
    "guidance_scale": 7.5
  }'
```

### Run the Locust benchmark
Open a second terminal (keep the server running) and start Locust from this directory:

```bash
locust -f locustfile.py
```

Headless example (no UI), for 10 users, spawn rate 2 users/sec, duration 5 minutes, and CSV output:

```bash
locust -f locustfile.py --host http://localhost:8000 \
  --headless -u 10 -r 2 -t 5m --csv fastapi_bench
```