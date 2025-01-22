# This script is adapted from the Greg Kamradt's notebook on chunking.
# Original code can be found at: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

from typing import Optional, List, Any
import numpy as np
from .base_chunker import BaseChunker
from .recursive_token_chunker import RecursiveTokenChunker
from .utils import openai_token_count, get_openai_embedding_function

class KamradtModifiedChunker(BaseChunker):
    """
    A chunker that splits text into chunks of approximately a specified average size based on semantic similarity.

    This was adapted from Greg Kamradt's notebook on chunking but with the modification of including an 
    average chunk size parameter. The original code can be found at: 
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    """
    
    def __init__(
        self, 
        avg_chunk_size: int = 400, 
        min_chunk_size: int = 50, 
        embedding_function: Optional[Any] = None, 
        length_function = openai_token_count
    ):
        """
        Initialize the chunker.
        
        Args:
            avg_chunk_size: Desired average chunk size in tokens
            min_chunk_size: Minimum chunk size in tokens
            embedding_function: Function to generate embeddings. Uses OpenAI by default
            length_function: Function to count tokens. Uses OpenAI tiktoken by default
        """
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=length_function
        )
        
        self.avg_chunk_size = avg_chunk_size
        self.embedding_function = embedding_function or get_openai_embedding_function()
        self.length_function = length_function

    def combine_sentences(self, sentences: List[dict], buffer_size: int = 1) -> List[dict]:
        """Combine sentences with a buffer for context."""
        for i in range(len(sentences)):
            combined_sentence = ''
            
            # Add preceding sentences
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '

            # Add current sentence
            combined_sentence += sentences[i]['sentence']

            # Add following sentences
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']

            sentences[i]['combined_sentence'] = combined_sentence

        return sentences

    def calculate_cosine_distances(self, sentences: List[dict]) -> tuple[List[float], List[dict]]:
        """Calculate cosine distances between sentence embeddings."""
        BATCH_SIZE = 500
        distances = []
        embedding_matrix = None

        # Process embeddings in batches
        for i in range(0, len(sentences), BATCH_SIZE):
            batch_sentences = sentences[i:i+BATCH_SIZE]
            batch_sentences = [sentence['combined_sentence'] for sentence in batch_sentences]
            embeddings = self.embedding_function.get_embeddings(batch_sentences)

            batch_embedding_matrix = np.array(embeddings)

            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        # Normalize vectors
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = embedding_matrix / norms

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
        
        # Calculate distances
        for i in range(len(sentences) - 1):
            similarity = similarity_matrix[i, i + 1]
            distance = 1 - similarity
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance

        return distances, sentences

    def split_text(self, text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        # Initial sentence splitting
        sentences_strips = self.splitter.split_text(text)
        sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences_strips)]

        # Combine sentences with context
        sentences = self.combine_sentences(sentences, 3)

        # Calculate semantic distances
        distances, sentences = self.calculate_cosine_distances(sentences)

        # Calculate optimal threshold using binary search
        total_tokens = sum(self.length_function(sentence['sentence']) for sentence in sentences)
        number_of_cuts = total_tokens // self.avg_chunk_size

        lower_limit = 0.0
        upper_limit = 1.0
        distances_np = np.array(distances)

        while upper_limit - lower_limit > 1e-6:
            threshold = (upper_limit + lower_limit) / 2.0
            num_points_above_threshold = np.sum(distances_np > threshold)
            
            if num_points_above_threshold > number_of_cuts:
                lower_limit = threshold
            else:
                upper_limit = threshold

        # Find split points and create chunks
        indices_above_thresh = [i for i, x in enumerate(distances) if x > threshold]
        chunks = []
        start_index = 0

        for index in indices_above_thresh:
            group = sentences[start_index:index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1

        # Add remaining text as final chunk
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks