from .base_embedder import EmbeddingManager
from .character_length_function import character_count

# Register the custom character count function
EmbeddingManager.register_token_counter("character", character_count)

