from .base_chunker import BaseChunker
from typing import List
import numpy as np
from .utils import EmbeddingManager
from .recursive_token_chunker import RecursiveTokenChunker

class ClusterSemanticChunker(BaseChunker):
    def __init__(self, 
                 embedding_function=None, 
                 max_chunk_size=400,
                 min_chunk_size=50,
                 length_function=None):
        
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=length_function or EmbeddingManager.get_token_counter(),
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        if isinstance(embedding_function, str):
            self.embedding_function = EmbeddingManager.get_embedder(embedding_function)
        else:
            self.embedding_function = embedding_function or EmbeddingManager.get_embedder()
        
        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size
        self.length_function = length_function or EmbeddingManager.get_token_counter()

    def _get_similarity_matrix(self, sentences):
        BATCH_SIZE = 500
        embedding_matrix = None

        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i:i+BATCH_SIZE]
            embeddings = self.embedding_function.get_embeddings(batch)
            batch_matrix = np.array(embeddings)
            
            if embedding_matrix is None:
                embedding_matrix = batch_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_matrix), axis=0)

        return np.dot(embedding_matrix, embedding_matrix.T)

    def _calculate_reward(self, matrix, start, end):
        return np.sum(matrix[start:end+1, start:end+1])

    def _optimal_segmentation(self, matrix, max_cluster_size):
        matrix = matrix - np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        np.fill_diagonal(matrix, 0)
        
        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
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
        sentences = self.splitter.split_text(text)
        if not sentences:
            return []
            
        similarity_matrix = self._get_similarity_matrix(sentences)
        clusters = self._optimal_segmentation(similarity_matrix, self.max_cluster)
        return [' '.join(sentences[start:end+1]) for start, end in clusters]