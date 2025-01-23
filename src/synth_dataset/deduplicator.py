import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class QuestionDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def _calculate_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """Calculate the cosine similarity matrix for the embeddings."""
        embeddings_matrix = np.array(embeddings)
        # Normalize the vectors
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        embeddings_matrix = embeddings_matrix / norms
        return np.dot(embeddings_matrix, embeddings_matrix.T)
    
    def _filter_similar_questions(self, similarity_matrix: np.ndarray) -> List[int]:
        """
        Filter out questions that are too similar to others.
        Returns indices of questions to keep.
        """
        n = similarity_matrix.shape[0]
        keep_mask = np.ones(n, dtype=bool)
        
        # For each question
        for i in range(n):
            if keep_mask[i]:
                # Find questions that are too similar and remove them
                similar_questions = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
                # Only look at questions after the current one
                similar_questions = similar_questions[similar_questions > i]
                keep_mask[similar_questions] = False
        
        return np.where(keep_mask)[0]
    
    def deduplicate(self, questions: List[Dict], embeddings: List[List[float]]) -> List[Dict]:
        """
        Remove duplicate and similar questions based on embedding similarity.
        
        Args:
            questions: List of question dictionaries
            embeddings: List of embedding vectors for questions
            
        Returns:
            List of filtered question dictionaries
        """
        if not questions:
            return []
            
        # First remove exact duplicates based on question text
        seen_questions = set()
        unique_questions = []
        unique_embeddings = []
        
        for q, emb in zip(questions, embeddings):
            if q["question"] not in seen_questions:
                seen_questions.add(q["question"])
                unique_questions.append(q)
                unique_embeddings.append(emb)
        
        if not unique_questions:
            return []
            
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(unique_embeddings)
        
        # Get indices of questions to keep
        keep_indices = self._filter_similar_questions(similarity_matrix)
        
        # Filter questions
        filtered_questions = [unique_questions[i] for i in keep_indices]
        
        logger.info(
            f"Deduplication: {len(questions)} -> {len(unique_questions)} -> "
            f"{len(filtered_questions)} questions after filtering"
        )
        return filtered_questions