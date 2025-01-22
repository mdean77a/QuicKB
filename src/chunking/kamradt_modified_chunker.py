
# This script is adapted from the Greg Kamradt's notebook on chunking.
# Original code can be found at: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
# chunking_evaluation modification: https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/kamradt_modified_chunker.py

from typing import Optional, List, Any
import numpy as np
from .base_chunker import BaseChunker
from .recursive_token_chunker import RecursiveTokenChunker
from embeddings.base_embedder import EmbeddingManager
from .registry import ChunkerRegistry 


@ChunkerRegistry.register("KamradtModifiedChunker")
class KamradtModifiedChunker(BaseChunker):
    def __init__(
        self,
        avg_chunk_size: int = 400,
        min_chunk_size: int = 50,
        embedding_function: Optional[Any] = None,
        length_function=None
    ):
        super().__init__(encoding_name="cl100k_base", length_function=length_function)

        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=self.length_function
        )

        if isinstance(embedding_function, str):
            self._embedding_function = EmbeddingManager.get_embedder(embedding_function)
        else:
            self._embedding_function = embedding_function or EmbeddingManager.get_embedder()

        self.avg_chunk_size = avg_chunk_size

    def combine_sentences(self, sentences: List[dict], buffer_size: int = 1) -> List[dict]:
        for i in range(len(sentences)):
            combined = []
            for j in range(max(0, i - buffer_size), min(len(sentences), i + buffer_size + 1)):
                combined.append(sentences[j]['sentence'])
            sentences[i]['combined_sentence'] = ' '.join(combined)
        return sentences

    def calculate_cosine_distances(self, sentences: List[dict]):
        embeddings = []
        for i in range(0, len(sentences), 500):
            batch = [s['combined_sentence'] for s in sentences[i:i + 500]]
            embeddings.extend(self._embedding_function.get_embeddings(batch))

        embedding_matrix = np.array(embeddings)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix /= norms
        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        distances = []
        for i in range(len(sentences) - 1):
            distance = 1 - similarity_matrix[i, i + 1]
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance
        return distances, sentences

    def split_text(self, text: str) -> List[str]:
        s_list = self.splitter.split_text(text)
        sentences = [{'sentence': s, 'index': i} for i, s in enumerate(s_list)]
        if not sentences:
            return []

        sentences = self.combine_sentences(sentences, 3)
        distances, sentences = self.calculate_cosine_distances(sentences)

        total_tokens = sum(self.length_function(s['sentence']) for s in sentences)
        target_splits = total_tokens // self.avg_chunk_size if self.avg_chunk_size else 1
        distances = np.array(distances)

        low, high = 0.0, 1.0
        while high - low > 1e-6:
            mid = (low + high) / 2
            if (distances > mid).sum() > target_splits:
                low = mid
            else:
                high = mid

        split_indices = [i for i, d in enumerate(distances) if d > high]
        chunks = []
        start = 0

        for idx in split_indices:
            chunks.append(' '.join(s['sentence'] for s in sentences[start:idx + 1]))
            start = idx + 1

        if start < len(sentences):
            chunks.append(' '.join(s['sentence'] for s in sentences[start:]))

        return chunks