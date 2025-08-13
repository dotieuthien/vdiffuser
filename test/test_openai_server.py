from openai import AsyncOpenAI
import base64
import asyncio
import time


client = AsyncOpenAI(
    base_url="http://localhost:8088/v1",
    api_key="vdiffuser_secret_key",
)

prompt = """
A children's book drawing of a veterinarian using a stethoscope to 
listen to the heartbeat of a baby otter.
"""

# Original single request test
async def test_single_request():
    result = await client.images.generate(
        model="auto",
        prompt=prompt
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    # Save the image to a file
    with open("test.png", "wb") as f:
        f.write(image_bytes)
    print("Single request completed - saved as test.png")


# New concurrent request test
async def generate_single_image(request_id):
    """Generate a single image and return the result with request ID"""
    start_time = time.time()
    
    result = await client.images.generate(
        model="auto",
        prompt=f"{prompt} (Request {request_id})"
    )
    
    end_time = time.time()
    
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    
    # Save with unique filename
    filename = f"test_concurrent_{request_id}.png"
    with open(filename, "wb") as f:
        f.write(image_bytes)
    
    return {
        "request_id": request_id,
        "filename": filename,
        "duration": end_time - start_time
    }


async def test_concurrent_requests():
    """Test concurrent image generation requests"""
    print("Starting concurrent image generation requests...")
    start_time = time.time()
    
    # Create concurrent tasks
    tasks = [generate_single_image(i+1) for i in range(2)]
    
    # Wait for all requests to complete concurrently
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Request failed with error: {result}")
            else:
                successful_results.append(result)
                print(f"Request {result['request_id']} completed in {result['duration']:.2f}s - saved as {result['filename']}")
        
        total_time = time.time() - start_time
        print(f"\nAll {len(successful_results)} concurrent requests completed in {total_time:.2f}s total")
        if successful_results:
            print(f"Average time per request: {total_time/len(successful_results):.2f}s")
        
        return successful_results
        
    except Exception as e:
        print(f"Error in concurrent requests: {e}")
        return []


async def main():
    # # Run single request test
    # print("=== Testing Single Request ===")
    # await test_single_request()
    
    print("\n=== Testing Concurrent Requests ===")
    # Run concurrent request test
    await test_concurrent_requests()


if __name__ == "__main__":
    asyncio.run(main())