from abc import ABC, abstractmethod
from typing import List, Any

class BaseChunker(ABC):
    def __init__(self, *args, **kwargs):
        self._resolve_length_function(kwargs)
        
    def _resolve_length_function(self, kwargs):
        """Common length function resolution"""
        encoding_name = kwargs.pop('encoding_name', "cl100k_base")
        length_func = kwargs.pop('length_function', None)
        
        from .utils import EmbeddingManager
        self.length_function = EmbeddingManager.resolve_length_function(
            length_func, encoding_name
        )

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass