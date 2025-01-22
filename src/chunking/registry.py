class ChunkerRegistry:
    _chunkers = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(chunker_class):
            cls._chunkers[name] = chunker_class
            return chunker_class
        return decorator
    
    @classmethod
    def get_chunker(cls, name: str):
        if name not in cls._chunkers:
            available = list(cls._chunkers.keys())
            raise ValueError(f"Unknown chunker: {name}. Available chunkers: {available}")
        return cls._chunkers[name]