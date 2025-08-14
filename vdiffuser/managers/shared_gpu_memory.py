import torch
from typing import Dict, Optional, Tuple

# Global storage for shared tensors with their metadata
_shared_tensors: Dict[str, Dict] = {}


def create_shared_tensor(name: str, tensor: torch.Tensor) -> bool:
    """Create/store a shared tensor with its shape info"""
    global _shared_tensors
    try:
        # Make tensor contiguous and share it
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        shared_tensor = tensor.share_memory_()
        
        # Store tensor with metadata
        _shared_tensors[name] = {
            'tensor': shared_tensor,
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'device': tensor.device
        }
        print(f"Created shared tensor: {name}, shape: {tensor.shape}")
        return True
    except Exception as e:
        print(f"Failed to create shared tensor {name}: {e}")
        return False


def read_shared_tensor(name: str) -> Optional[torch.Tensor]:
    """Read/get a shared tensor"""
    global _shared_tensors
    if name not in _shared_tensors:
        return None
    
    tensor_info = _shared_tensors[name]
    tensor = tensor_info['tensor']
    
    # Ensure tensor has correct shape
    if tensor.shape != tensor_info['shape']:
        tensor = tensor.view(tensor_info['shape'])
    
    return tensor


def update_shared_tensor(name: str, tensor: torch.Tensor) -> bool:
    """Update an existing shared tensor"""
    global _shared_tensors
    if name not in _shared_tensors:
        print(f"Tensor {name} not found for update")
        return False
    
    try:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        shared_tensor = tensor.share_memory_()
        
        # Update tensor with metadata
        _shared_tensors[name] = {
            'tensor': shared_tensor,
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'device': tensor.device
        }
        print(f"Updated shared tensor: {name}, shape: {tensor.shape}")
        return True
    except Exception as e:
        print(f"Failed to update shared tensor {name}: {e}")
        return False


def delete_shared_tensor(name: str) -> bool:
    """Delete a shared tensor"""
    global _shared_tensors
    if name in _shared_tensors:
        del _shared_tensors[name]
        print(f"Deleted shared tensor: {name}")
        return True
    else:
        print(f"Tensor {name} not found for deletion")
        return False


def get_tensor_info(name: str) -> Optional[Dict]:
    """Get tensor metadata (shape, dtype, device)"""
    global _shared_tensors
    if name not in _shared_tensors:
        return None
    
    info = _shared_tensors[name]
    return {
        'shape': info['shape'],
        'dtype': info['dtype'],
        'device': info['device']
    }


def list_shared_tensors() -> list:
    """List all shared tensor names"""
    global _shared_tensors
    return list(_shared_tensors.keys())


def clear_all_tensors():
    """Clear all shared tensors"""
    global _shared_tensors
    _shared_tensors.clear()
    print("Cleared all shared tensors")
