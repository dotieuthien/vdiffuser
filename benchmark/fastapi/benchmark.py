import requests
import json
import time
from typing import Dict, Any

SERVER_URL = "http://localhost:8000"
BENCHMARK_URL = f"{SERVER_URL}/benchmark"

def check_health() -> bool:
    try:
        response = requests.get(f"{BENCHMARK_URL}/health")
        response.raise_for_status()
        health_data = response.json()
        print(f"Health check: {health_data}")
        return health_data.get("pipeline_loaded", False)
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def benchmark_generation(prompt: str, **kwargs) -> Dict[str, Any]:
    
    payload = {
        "prompt": prompt,
        **kwargs
    }
    
    print(f"Sending request: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(f"{BENCHMARK_URL}/generate", json=payload)
        response.raise_for_status()
        request_time = time.time() - start_time
        
        result = response.json()
        result["total_request_time"] = round(request_time, 3)
        
        print(f"Response: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        print(f"Generation request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error response: {e.response.text}")
        return {}

def run_benchmark_suite():    
    print("=== FastAPI Diffusion Benchmark Client ===\n")
    
    if not check_health():
        print("Server is not ready. Please ensure the server is running and the pipeline is loaded.")
        return
    
    print("\n=== Running Benchmark Suite ===\n")
    
    test_cases = [
        {
            "name": "Quick Test (Low Steps)",
            "prompt": "a cute cat sitting on a table",
            "num_inference_steps": 10,
            "width": 512,
            "height": 512,
        },
        {
            "name": "Standard Quality",
            "prompt": "a beautiful landscape with mountains and lakes at sunset",
            "num_inference_steps": 20,
            "width": 1024,
            "height": 1024,
        },
        {
            "name": "High Quality",
            "prompt": "a detailed portrait of a majestic lion",
            "num_inference_steps": 30,
            "width": 1024,
            "height": 1024,
            "guidance_scale": 8.0,
        },
        {
            "name": "With Negative Prompt",
            "prompt": "a modern city skyline at night",
            "negative_prompt": "blurry, low quality, distorted",
            "num_inference_steps": 20,
            "width": 1024,
            "height": 1024,
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        name = test_case.pop("name")
        print(f"\n--- Test {i}: {name} ---")
        
        result = benchmark_generation(**test_case)
        if result:
            results.append({"test_name": name, **result})
            print(f"Generation time: {result.get('time_s', 'N/A')}s")
            print(f"Memory usage: {result.get('mem_gb', 'N/A')} GB")
            print(f"Total request time: {result.get('total_request_time', 'N/A')}s")
        
        time.sleep(1)
    
    print("\n=== Benchmark Summary ===")
    for result in results:
        print(f"{result['test_name']}: {result.get('time_s', 'N/A')}s (gen) | "
              f"{result.get('mem_gb', 'N/A')} GB (mem)")

if __name__ == "__main__":
    run_benchmark_suite()
