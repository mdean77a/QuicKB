from abc import ABC, abstractmethod
from typing import List, Any

# Correct the import for EmbeddingManager
from embeddings.base_embedder import EmbeddingManager

class BaseChunker(ABC):
    def __init__(self, *args, **kwargs):
        self._resolve_length_function(kwargs)

    def _resolve_length_function(self, kwargs):
        """Common length function resolution."""
        encoding_name = kwargs.pop('encoding_name', "cl100k_base")
        length_func = kwargs.pop('length_function', None)

        # Use the corrected import path for EmbeddingManager:
        self.length_function = length_func or EmbeddingManager.get_token_counter(encoding_name)

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass
