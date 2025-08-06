from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union


class EngineBase(ABC):
    """
    Abstract base class for engine interfaces that support generation, and memory control.
    This base class provides a unified API for both HTTP-based engines and engines.
    """

    @abstractmethod
    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        image_data: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        stream: Optional[bool] = None,
        data_parallel_rank: Optional[int] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """Generate outputs based on given inputs."""
        pass

    @abstractmethod
    def flush_cache(self):
        """Flush the cache of the engine."""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown the engine and clean up resources."""
        pass