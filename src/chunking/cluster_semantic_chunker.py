
# This script is adapted from the chunking_evaluation package, developed by ChromaDB Research.
# Original code can be found at: https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/cluster_semantic_chunker.py
# License: MIT License

from .base_chunker import BaseChunker
from typing import List, Optional
import numpy as np
from litellm import embedding
from .recursive_token_chunker import RecursiveTokenChunker
from .registry import ChunkerRegistry
from .utils import get_length_function
import logging

logger = logging.getLogger(__name__)

@ChunkerRegistry.register("ClusterSemanticChunker")
class ClusterSemanticChunker(BaseChunker):
    def __init__(
        self,
        max_chunk_size: int = 400,
        min_chunk_size: int = 50,
        length_type: str = 'token',
        litellm_config: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(length_type=length_type, **kwargs)
        
        self.length_function = get_length_function(
            length_type=length_type,
            encoding_name=kwargs.get('encoding_name', 'cl100k_base'),
            model_name=kwargs.get('model_name')
        )

        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=self.length_function,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        self._litellm_config = litellm_config or {}
        self.max_cluster = max_chunk_size // min_chunk_size

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Universal embedding response handler"""
        try:
            response = embedding(
                model=self._litellm_config.get('embedding_model', 'text-embedding-3-large'),
                input=texts,
                api_base=self._litellm_config.get('embedding_api_base')
            )
            
            # Handle all possible response formats
            if isinstance(response, dict):
                items = response.get('data', [])
            elif hasattr(response, 'data'):
                items = response.data
            else:
                items = response
                
            return [
                item['embedding'] if isinstance(item, dict) else item.embedding
                for item in items
            ]
            
        except Exception as e:
            logger.error(f"Embedding failed for batch: {str(e)}")
            raise RuntimeError(f"Embedding error: {str(e)}")

    def _calculate_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Batch processing with error logging"""
        if not sentences:
            return np.zeros((0, 0))
            
        embeddings = []
        for batch_idx in range(0, len(sentences), 500):
            try:
                batch = sentences[batch_idx:batch_idx+500]
                embeddings.extend(self._get_embeddings(batch))
            except Exception as e:
                logger.error(f"Failed processing batch {batch_idx//500}: {str(e)}")
                raise
                
        embedding_matrix = np.array(embeddings)
        return np.dot(embedding_matrix, embedding_matrix.T)

    def _optimal_segmentation(self, matrix: np.ndarray) -> List[tuple]:
        """Original Chroma algorithm implementation"""
        n = matrix.shape[0]
        if n < 1:
            return []

        # Calculate mean of off-diagonal elements
        triu = np.triu_indices(n, k=1)
        tril = np.tril_indices(n, k=-1)
        mean_value = (matrix[triu].sum() + matrix[tril].sum()) / (n * (n - 1)) if n > 1 else 0
        
        matrix = matrix - mean_value
        np.fill_diagonal(matrix, 0)

        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, min(self.max_cluster + 1, i + 2)):
                start_idx = i - size + 1
                if start_idx >= 0:
                    current_reward = matrix[start_idx:i+1, start_idx:i+1].sum()
                    if start_idx > 0:
                        current_reward += dp[start_idx - 1]
                    if current_reward > dp[i]:
                        dp[i] = current_reward
                        segmentation[i] = start_idx

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        return list(reversed(clusters))

    def split_text(self, text: str) -> List[str]:
        """Main processing pipeline"""
        if not text.strip():
            return []

        # First-stage splitting
        sentences = self.splitter.split_text(text)
        if len(sentences) < 2:
            return [text]

        # Semantic clustering
        similarity_matrix = self._calculate_similarity_matrix(sentences)
        clusters = self._optimal_segmentation(similarity_matrix)
        
        return [' '.join(sentences[start:end+1]) for start, end in clusters]