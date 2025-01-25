
# This script is adapted from the chunking_evaluation package, developed by ChromaDB Research.
# Original code can be found at: https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/cluster_semantic_chunker.py
# License: MIT License

from .base_chunker import BaseChunker
from typing import List
import numpy as np
from litellm import embedding
from .recursive_token_chunker import RecursiveTokenChunker
from .registry import ChunkerRegistry

@ChunkerRegistry.register("ClusterSemanticChunker")
class ClusterSemanticChunker(BaseChunker):
    def __init__(
        self,
        litellm_config=None,
        max_chunk_size=400,
        min_chunk_size=50,
        length_type='token',
        **kwargs
    ):
        super().__init__(length_type=length_type, **kwargs)

        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=self.length_function,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        self._litellm_config = litellm_config or {}
        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size

    def _get_embeddings(self, texts):
        """Get embeddings using LiteLLM."""
        try:
            response = embedding(
                model=self._litellm_config.get('embedding_model', 'text-embedding-3-large'),
                input=texts,
                api_base=self._litellm_config.get('embedding_api_base')
            )
            # Extract embeddings from LiteLLM response
            # Response format is typically: {'data': [{'embedding': [...], 'index': 0}, ...]}
            if hasattr(response, 'data'):
                # Handle response object
                return [item['embedding'] for item in response.data]
            elif isinstance(response, dict) and 'data' in response:
                # Handle response dict
                return [item['embedding'] for item in response['data']]
            else:
                raise ValueError(f"Unexpected response format from LiteLLM: {response}")
        except Exception as e:
            raise RuntimeError(f"Error getting embeddings: {str(e)}")

    def _get_similarity_matrix(self, sentences):
        if not sentences:
            return np.array([[]])
            
        BATCH_SIZE = 500
        embedding_matrix = None

        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i:i + BATCH_SIZE]
            embeddings = self._get_embeddings(batch)
            batch_matrix = np.array(embeddings)

            if embedding_matrix is None:
                embedding_matrix = batch_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_matrix), axis=0)

        # Normalize the embeddings for cosine similarity
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = embedding_matrix / (norms + 1e-8)  # Add small epsilon to avoid division by zero
        
        return np.dot(embedding_matrix, embedding_matrix.T)

    def _calculate_reward(self, matrix, start, end):
        if start > end or start < 0 or end >= matrix.shape[0]:
            return 0
        return np.sum(matrix[start:end + 1, start:end + 1])

    def _optimal_segmentation(self, matrix, max_cluster_size):
        if matrix.size == 0:
            return []
            
        # Adjust matrix by subtracting average off-diagonal
        matrix = matrix - np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        np.fill_diagonal(matrix, 0)

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, min(max_cluster_size + 1, i + 2)):
                if i - size + 1 >= 0:
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    if i - size >= 0:
                        reward += dp[i - size]
                    if reward > dp[i]:
                        dp[i] = reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        return list(reversed(clusters))

    def split_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
            
        sentences = self.splitter.split_text(text)
        if not sentences:
            return []

        similarity_matrix = self._get_similarity_matrix(sentences)
        clusters = self._optimal_segmentation(similarity_matrix, self.max_cluster)
        return [' '.join(sentences[start:end + 1]) for start, end in clusters]