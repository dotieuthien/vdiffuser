import torch
import torch.multiprocessing as mp
from typing import Dict, Optional, Tuple

# Remove global storage - now use shared dictionary passed as parameter

def create_shared_dict() -> Dict:
    """Create a shared dictionary for multi-process tensor sharing"""
    mp.set_start_method("spawn", force=True)  # Safe start method
    manager = mp.Manager()
    return manager.dict()


def create_shared_tensor(shared_dict: Dict, name: str, tensor: torch.Tensor) -> bool:
    """Create/store a shared tensor with its shape info"""
    try:
        # Make tensor contiguous and share it
        # Detach tensor to remove gradients (required for cross-process serialization)
        tensor = tensor.detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        shared_tensor = tensor.share_memory_()
        
        # Store tensor with metadata
        shared_dict[name] = {
            'tensor': shared_tensor,
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'device': tensor.device
        }
        print(f"[{mp.current_process().name}] Created shared tensor: {name}, shape: {tensor.shape}")
        return True
    except Exception as e:
        print(f"[{mp.current_process().name}] Failed to create shared tensor {name}: {e}")
        return False


def create_shared_tensor_tuple(shared_dict: Dict, name: str, tensor_tuple: tuple) -> bool:
    """Create/store a tuple of shared tensors with their metadata"""
    try:
        shared_tensors_info = []
        
        for i, tensor in enumerate(tensor_tuple):
            if isinstance(tensor, torch.Tensor):
                # Make tensor contiguous and share it
                tensor = tensor.detach().cpu()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                shared_tensor = tensor.share_memory_()
                
                # Store tensor info
                tensor_info = {
                    'tensor': shared_tensor,
                    'shape': tensor.shape,
                    'dtype': tensor.dtype,
                    'device': tensor.device,
                    'index': i
                }
                shared_tensors_info.append(tensor_info)
            else:
                # Handle non-tensor items in tuple
                shared_tensors_info.append({
                    'tensor': tensor,  # Store as-is for non-tensors
                    'shape': None,
                    'dtype': None,
                    'device': None,
                    'index': i,
                    'is_tensor': False
                })
        
        # Store tuple metadata
        shared_dict[name] = {
            'tuple_length': len(tensor_tuple),
            'tensors': shared_tensors_info,
            'is_tuple': True
        }
        
        print(f"[{mp.current_process().name}] Created shared tensor tuple: {name}, length: {len(tensor_tuple)}")
        return True
    except Exception as e:
        print(f"[{mp.current_process().name}] Failed to create shared tensor tuple {name}: {e}")
        return False


def read_shared_tensor(shared_dict: Dict, name: str) -> Optional[torch.Tensor]:
    """Read/get a shared tensor"""
    if name not in shared_dict:
        return None
    
    tensor_info = shared_dict[name]
    tensor = tensor_info['tensor']
    
    # Ensure tensor has correct shape
    if tensor.shape != tensor_info['shape']:
        tensor = tensor.view(tensor_info['shape'])
    
    return tensor


def read_shared_tensor_tuple(shared_dict: Dict, name: str) -> Optional[tuple]:
    """Read/get a tuple of shared tensors"""
    if name not in shared_dict:
        return None
    
    tuple_info = shared_dict[name]
    
    # Check if it's actually a tuple
    if not tuple_info.get('is_tuple', False):
        print(f"[{mp.current_process().name}] {name} is not a tensor tuple")
        return None
    
    tensors = []
    
    for tensor_info in tuple_info['tensors']:
        if tensor_info.get('is_tensor', True):  # Default to True for backward compatibility
            tensor = tensor_info['tensor']
            
            # Ensure tensor has correct shape
            if tensor_info['shape'] is not None and tensor.shape != tensor_info['shape']:
                tensor = tensor.view(tensor_info['shape'])
            
            tensors.append(tensor)
        else:
            # Non-tensor item
            tensors.append(tensor_info['tensor'])
    
    return tuple(tensors)


def update_shared_tensor(shared_dict: Dict, name: str, tensor: torch.Tensor) -> bool:
    """Update an existing shared tensor"""
    if name not in shared_dict:
        print(f"[{mp.current_process().name}] Tensor {name} not found for update")
        return False
    
    try:
        # Detach tensor to remove gradients (required for cross-process serialization)
        tensor = tensor.detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        shared_tensor = tensor.share_memory_()
        
        # Update tensor with metadata
        shared_dict[name] = {
            'tensor': shared_tensor,
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'device': tensor.device
        }
        print(f"[{mp.current_process().name}] Updated shared tensor: {name}, shape: {tensor.shape}")
        return True
    except Exception as e:
        print(f"[{mp.current_process().name}] Failed to update shared tensor {name}: {e}")
        return False


def delete_shared_tensor(shared_dict: Dict, name: str) -> bool:
    """Delete a shared tensor"""
    if name in shared_dict:
        del shared_dict[name]
        print(f"[{mp.current_process().name}] Deleted shared tensor: {name}")
        return True
    else:
        print(f"[{mp.current_process().name}] Tensor {name} not found for deletion")
        return False


def get_tensor_info(shared_dict: Dict, name: str) -> Optional[Dict]:
    """Get tensor metadata (shape, dtype, device)"""
    if name not in shared_dict:
        return None
    
    info = shared_dict[name]
    return {
        'shape': info['shape'],
        'dtype': info['dtype'],
        'device': info['device']
    }


def list_shared_tensors(shared_dict: Dict) -> list:
    """List all shared tensor names"""
    return list(shared_dict.keys())


def clear_all_tensors(shared_dict: Dict):
    """Clear all shared tensors"""
    shared_dict.clear()
    print(f"[{mp.current_process().name}] Cleared all shared tensors")


def worker_example(shared_dict: Dict, tensor_name: str):
    """Example worker function to demonstrate usage"""
    tensor = read_shared_tensor(shared_dict, tensor_name)
    if tensor is not None:
        print(f"[{mp.current_process().name}] Read tensor: {tensor_name}, shape: {tensor.shape}, sum: {tensor.sum()}")
    else:
        print(f"[{mp.current_process().name}] Tensor {tensor_name} not found")


# Example usage
if __name__ == "__main__":
    # Create shared dictionary
    with mp.Manager() as manager:
        shared_dict = manager.dict()
        
        # Create a shared tensor in the main process
        tensor = torch.ones((3, 3)) * 5
        create_shared_tensor(shared_dict, "example_tensor", tensor)
        
        # Start another process that reads the same tensor
        process = mp.Process(target=worker_example, args=(shared_dict, "example_tensor"))
        process.start()
        process.join()
        
        print(f"[{mp.current_process().name}] Available tensors: {list_shared_tensors(shared_dict)}")
