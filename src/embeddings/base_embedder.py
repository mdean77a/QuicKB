from typing import Callable, Dict, Any, Optional

class EmbeddingManager:
    _custom_embedders: Dict[str, Any] = {}
    _custom_token_counters: Dict[str, Callable[[str], int]] = {}
    
    @classmethod
    def register_embedder(cls, name: str, embedder: object):
        cls._custom_embedders[name] = embedder
        
    @classmethod
    def register_token_counter(cls, name: str, counter: Callable[[str], int]):
        cls._custom_token_counters[name] = counter
    
    @classmethod
    def get_embedder(cls, name: Optional[str] = None) -> object:
        if name and name in cls._custom_embedders:
            return cls._custom_embedders[name]
        return cls.get_default_embedder()
    
    @classmethod
    def get_token_counter(cls, name: Optional[str] = None) -> Callable[[str], int]:
        if name and name in cls._custom_token_counters:
            return cls._custom_token_counters[name]
        return cls.get_default_token_counter()
    
    @staticmethod
    def get_default_embedder() -> object:
        from .openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()
    
    @staticmethod
    def get_default_token_counter() -> Callable[[str], int]:
        from chunking.utils import openai_token_count
        return openai_token_count