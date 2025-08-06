from typing import Dict, Union, Optional
import hashlib
import torch
from PIL import Image
import io


class ImageCache:
    """ImageCache is used to store diffusion model results at various pipeline stages"""

    def __init__(
        self,
        max_size: int,
    ):
        self.max_size = max_size

        # Separate caches for different pipeline stages
        self.text_embedding_cache: Dict[int, torch.Tensor] = {}
        self.latent_cache: Dict[int, torch.Tensor] = {}
        self.image_cache: Dict[int, Image.Image] = {}

        # Size tracking
        self.current_size = 0

        # Cache hit statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def put_text_embedding(self, prompt_hash: int, embedding: torch.Tensor) -> bool:
        """Store text embedding for a given prompt hash."""
        if prompt_hash in self.text_embedding_cache:
            return True

        data_size = self._get_tensor_size(embedding)
        if not self._check_and_make_space(data_size):
            return False

        self.text_embedding_cache[prompt_hash] = embedding.clone()
        self.current_size += data_size
        return True

    def put_latent(self, generation_hash: int, latent: torch.Tensor) -> bool:
        """Store latent tensor for a given generation configuration hash."""
        if generation_hash in self.latent_cache:
            return True

        data_size = self._get_tensor_size(latent)
        if not self._check_and_make_space(data_size):
            return False

        self.latent_cache[generation_hash] = latent.clone()
        self.current_size += data_size
        return True

    def put_image(self, image_hash: int, image: Image.Image) -> bool:
        """Store final generated image."""
        if image_hash in self.image_cache:
            return True

        data_size = self._get_image_size(image)
        if not self._check_and_make_space(data_size):
            return False

        # Store a copy of the image
        self.image_cache[image_hash] = image.copy()
        self.current_size += data_size
        return True

    def has_text_embedding(self, prompt_hash: int) -> bool:
        """Check if text embedding exists in cache."""
        return prompt_hash in self.text_embedding_cache

    def has_latent(self, generation_hash: int) -> bool:
        """Check if latent exists in cache."""
        return generation_hash in self.latent_cache

    def has_image(self, image_hash: int) -> bool:
        """Check if image exists in cache."""
        return image_hash in self.image_cache

    def get_text_embedding(self, prompt_hash: int) -> Optional[torch.Tensor]:
        """Retrieve text embedding from cache."""
        if prompt_hash in self.text_embedding_cache:
            self.cache_hits += 1
            return self.text_embedding_cache[prompt_hash].clone()
        self.cache_misses += 1
        return None

    def get_latent(self, generation_hash: int) -> Optional[torch.Tensor]:
        """Retrieve latent from cache."""
        if generation_hash in self.latent_cache:
            self.cache_hits += 1
            return self.latent_cache[generation_hash].clone()
        self.cache_misses += 1
        return None

    def get_image(self, image_hash: int) -> Optional[Image.Image]:
        """Retrieve image from cache."""
        if image_hash in self.image_cache:
            self.cache_hits += 1
            return self.image_cache[image_hash].copy()
        self.cache_misses += 1
        return None

    def free_text_embedding(self, prompt_hash: int) -> bool:
        """Remove text embedding from cache."""
        if prompt_hash not in self.text_embedding_cache:
            return False
        old_embedding = self.text_embedding_cache.pop(prompt_hash)
        self.current_size -= self._get_tensor_size(old_embedding)
        return True

    def free_latent(self, generation_hash: int) -> bool:
        """Remove latent from cache."""
        if generation_hash not in self.latent_cache:
            return False
        old_latent = self.latent_cache.pop(generation_hash)
        self.current_size -= self._get_tensor_size(old_latent)
        return True

    def free_image(self, image_hash: int) -> bool:
        """Remove image from cache."""
        if image_hash not in self.image_cache:
            return False
        old_image = self.image_cache.pop(image_hash)
        self.current_size -= self._get_image_size(old_image)
        return True

    def clear(self):
        """Clear all caches."""
        self.text_embedding_cache.clear()
        self.latent_cache.clear()
        self.image_cache.clear()
        self.current_size = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def clear_text_embeddings(self):
        """Clear only text embedding cache."""
        for embedding in self.text_embedding_cache.values():
            self.current_size -= self._get_tensor_size(embedding)
        self.text_embedding_cache.clear()

    def clear_latents(self):
        """Clear only latent cache."""
        for latent in self.latent_cache.values():
            self.current_size -= self._get_tensor_size(latent)
        self.latent_cache.clear()

    def clear_images(self):
        """Clear only image cache."""
        for image in self.image_cache.values():
            self.current_size -= self._get_image_size(image)
        self.image_cache.clear()

    def _check_and_make_space(self, required_size: int) -> bool:
        """Check if there's space and attempt to make space if needed."""
        if self.current_size + required_size <= self.max_size:
            return True

        # Try to evict items using LRU-like strategy
        # For simplicity, we'll clear oldest items by cache type
        return self._evict_to_make_space(required_size)

    def _evict_to_make_space(self, required_size: int) -> bool:
        """Evict items to make space. Simple FIFO strategy."""
        original_size = self.current_size

        # First, try to evict images (usually largest)
        if self.current_size + required_size > self.max_size and self.image_cache:
            items_to_remove = list(self.image_cache.keys())[
                :len(self.image_cache)//2]
            for key in items_to_remove:
                self.free_image(key)
                if self.current_size + required_size <= self.max_size:
                    return True

        # Then try latents
        if self.current_size + required_size > self.max_size and self.latent_cache:
            items_to_remove = list(self.latent_cache.keys())[
                :len(self.latent_cache)//2]
            for key in items_to_remove:
                self.free_latent(key)
                if self.current_size + required_size <= self.max_size:
                    return True

        # Finally, text embeddings (smallest, so keep longer)
        if self.current_size + required_size > self.max_size and self.text_embedding_cache:
            items_to_remove = list(self.text_embedding_cache.keys())[
                :len(self.text_embedding_cache)//2]
            for key in items_to_remove:
                self.free_text_embedding(key)
                if self.current_size + required_size <= self.max_size:
                    return True

        return self.current_size + required_size <= self.max_size

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Calculate tensor size in bytes."""
        return tensor.element_size() * tensor.numel()

    def _get_image_size(self, image: Image.Image) -> int:
        """Calculate image size in bytes."""
        # Estimate PIL Image memory usage
        width, height = image.size
        mode = image.mode

        # Bytes per pixel based on mode
        bytes_per_pixel = {
            'L': 1,      # Grayscale
            'RGB': 3,    # RGB
            'RGBA': 4,   # RGBA
            'CMYK': 4,   # CMYK
            'P': 1,      # Palette
        }.get(mode, 3)  # Default to RGB

        return width * height * bytes_per_pixel

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        total_items = len(self.text_embedding_cache) + \
            len(self.latent_cache) + len(self.image_cache)
        hit_rate = self.cache_hits / \
            max(1, self.cache_hits + self.cache_misses)

        return {
            'total_items': total_items,
            'text_embeddings': len(self.text_embedding_cache),
            'latents': len(self.latent_cache),
            'images': len(self.image_cache),
            'current_size_bytes': self.current_size,
            'max_size_bytes': self.max_size,
            'utilization': self.current_size / self.max_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
        }

    def __len__(self):
        """Total number of cached items across all caches."""
        return len(self.text_embedding_cache) + len(self.latent_cache) + len(self.image_cache)

    @staticmethod
    def hash_prompt(prompt: str, **kwargs) -> int:
        """Create hash for prompt and text encoding parameters."""
        # Include relevant text encoding parameters in hash
        hash_str = f"{prompt}"
        for key in sorted(kwargs.keys()):
            hash_str += f"_{key}:{kwargs[key]}"
        return int(hashlib.md5(hash_str.encode()).hexdigest()[:8], 16)

    @staticmethod
    def hash_generation_config(
        prompt_hash: int,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int] = None,
        **kwargs
    ) -> int:
        """Create hash for generation configuration."""
        config_str = f"{prompt_hash}_{width}x{height}_{num_inference_steps}_{guidance_scale}"
        if seed is not None:
            config_str += f"_{seed}"
        for key in sorted(kwargs.keys()):
            config_str += f"_{key}:{kwargs[key]}"
        return int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16)

    @staticmethod
    def hash_image_config(
        generation_hash: int,
        output_format: str = "PNG",
        **kwargs
    ) -> int:
        """Create hash for final image configuration."""
        config_str = f"{generation_hash}_{output_format}"
        for key in sorted(kwargs.keys()):
            config_str += f"_{key}:{kwargs[key]}"
        return int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16)
