#!/usr/bin/env python3

import torch
import sys
import os

# Add the vdiffuser package to Python path
sys.path.insert(0, '/root/vdiffuser')

from vdiffuser.managers.shared_gpu_memory import (
    create_shared_tensor_with_metadata,
    read_shared_tensor_with_metadata,
)

def test_tensor_serialization():
    """Test that tensor serialization and deserialization works correctly"""
    print("Testing tensor serialization...")
    
    # Create test tensors
    test_tensors = [
        torch.randn(2, 3, 4),  # 3D tensor
        torch.ones(5, 5),      # 2D tensor
        torch.zeros(10),       # 1D tensor
    ]
    
    # Test CPU tensors
    for i, tensor in enumerate(test_tensors):
        print(f"\nTesting tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # Create shared tensor metadata
        tensor_name = f"test_tensor_{i}"
        metadata = create_shared_tensor_with_metadata(tensor_name, tensor)
        
        if metadata is None:
            print(f"Failed to create metadata for tensor {i}")
            continue
            
        print(f"Created metadata: {metadata}")
        
        # Reconstruct tensor from metadata
        reconstructed = read_shared_tensor_with_metadata(f"new_{tensor_name}", metadata)
        
        if reconstructed is None:
            print(f"Failed to reconstruct tensor {i}")
            continue
            
        print(f"Reconstructed tensor: shape={reconstructed.shape}, dtype={reconstructed.dtype}")
        
        # Check if tensors are equal
        if torch.equal(tensor, reconstructed):
            print(f"✓ Tensor {i} reconstruction successful!")
        else:
            print(f"✗ Tensor {i} reconstruction failed - tensors are not equal")
            print(f"Original: {tensor.flatten()[:5]}...")
            print(f"Reconstructed: {reconstructed.flatten()[:5]}...")

    # Test GPU tensors if CUDA is available
    if torch.cuda.is_available():
        print(f"\nTesting GPU tensors...")
        gpu_tensor = torch.randn(3, 3).cuda()
        print(f"GPU tensor: shape={gpu_tensor.shape}, device={gpu_tensor.device}")
        
        metadata = create_shared_tensor_with_metadata("gpu_test", gpu_tensor)
        if metadata:
            # Reconstruct on CPU first (this is what happens in cross-process scenario)
            reconstructed_cpu = read_shared_tensor_with_metadata("gpu_test_reconstructed", metadata)
            if reconstructed_cpu is not None:
                # Move back to GPU
                reconstructed_gpu = reconstructed_cpu.cuda()
                
                if torch.equal(gpu_tensor.cpu(), reconstructed_cpu):
                    print("✓ GPU tensor reconstruction successful!")
                else:
                    print("✗ GPU tensor reconstruction failed")
            else:
                print("✗ Failed to reconstruct GPU tensor")
        else:
            print("✗ Failed to create GPU tensor metadata")
    else:
        print("\nCUDA not available, skipping GPU tests")

if __name__ == "__main__":
    test_tensor_serialization() 