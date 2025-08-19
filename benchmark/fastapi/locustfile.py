import json
import random
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner


class DiffusionBenchmarkUser(HttpUser):
    wait_time = between(1, 2)
    host = "http://localhost:8000"
    
    def on_start(self):
        print("Starting benchmark...")  
    
    @task(3)
    def quick_generation(self):
        prompts = [
            "a cute cat sitting on a table",
            "a beautiful flower in a garden", 
            "a simple landscape",
            "a red car on a road",
            "a peaceful beach scene"
        ]
        
        payload = {
            "prompt": random.choice(prompts),
            "num_inference_steps": 10,
            "width": 512,
            "height": 512,
            "guidance_scale": 7.5
        }
        
        with self.client.post("/benchmark/generate", 
                             json=payload,
                             catch_response=True,
                             name="quick_generation") as response:
            if response.status_code == 200:
                result = response.json()
                events.request.fire(
                    request_type="GENERATE", 
                    name="generation_time",
                    response_time=result.get("time_s", 0) * 1000,  # Convert to ms
                    response_length=0
                )
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def standard_quality_generation(self):
        prompts = [
            "a beautiful landscape with mountains and lakes at sunset",
            "a majestic castle on a hill",
            "a detailed city skyline at golden hour",
            "a serene forest path in autumn",
            "a modern architecture building"
        ]
        
        payload = {
            "prompt": random.choice(prompts),
            "num_inference_steps": 20,
            "width": 1024,
            "height": 1024,
            "guidance_scale": 7.5
        }
        
        with self.client.post("/benchmark/generate",
                             json=payload,
                             catch_response=True, 
                             name="standard_generation") as response:
            if response.status_code == 200:
                result = response.json()
                events.request.fire(
                    request_type="GENERATE",
                    name="generation_time", 
                    response_time=result.get("time_s", 0) * 1000,
                    response_length=0
                )
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def high_quality_generation(self):
        prompts = [
            "a detailed portrait of a majestic lion",
            "an intricate fantasy dragon",
            "a photorealistic human portrait",
            "a detailed mechanical robot",
            "an elaborate fantasy landscape"
        ]
        
        payload = {
            "prompt": random.choice(prompts),
            "num_inference_steps": 30,
            "width": 1024,
            "height": 1024,
            "guidance_scale": 8.0
        }
        
        with self.client.post("/benchmark/generate",
                             json=payload,
                             catch_response=True,
                             name="high_quality_generation") as response:
            if response.status_code == 200:
                result = response.json()
                events.request.fire(
                    request_type="GENERATE",
                    name="generation_time",
                    response_time=result.get("time_s", 0) * 1000,
                    response_length=0
                )
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def negative_prompt_generation(self):
        """Generation with negative prompts"""
        prompts = [
            "a modern city skyline at night",
            "a peaceful garden scene",
            "a vintage car on a road",
            "a cozy interior room",
            "a mountain lake reflection"
        ]
        
        negative_prompts = [
            "blurry, low quality, distorted",
            "dark, unclear, bad composition", 
            "pixelated, artifacts, noise",
            "oversaturated, unrealistic",
            "deformed, ugly, bad anatomy"
        ]
        
        payload = {
            "prompt": random.choice(prompts),
            "negative_prompt": random.choice(negative_prompts),
            "num_inference_steps": 20,
            "width": 1024,
            "height": 1024,
            "guidance_scale": 7.5
        }
        
        with self.client.post("/benchmark/generate",
                             json=payload,
                             catch_response=True,
                             name="negative_prompt_generation") as response:
            if response.status_code == 200:
                result = response.json()
                events.request.fire(
                    request_type="GENERATE",
                    name="generation_time",
                    response_time=result.get("time_s", 0) * 1000,
                    response_length=0
                )
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    if isinstance(environment.runner, MasterRunner):
        print("Running in Master mode")
    elif isinstance(environment.runner, WorkerRunner):
        print("Running in Worker mode")
    else:
        print("Running in standalone mode")


@events.test_start.add_listener 
def on_test_start(environment, **kwargs):
    print("Starting diffusion model benchmark...")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("\nBenchmark completed!")
    print("Check the Locust web UI for detailed results.")