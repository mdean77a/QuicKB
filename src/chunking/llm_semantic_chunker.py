
# This script is adapted from the chunking_evaluation package, developed by ChromaDB Research.
# Original code can be found at: https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/llm_semantic_chunker.py
# License: MIT License

from .base_chunker import BaseChunker
from .recursive_token_chunker import RecursiveTokenChunker
import backoff
from tqdm import tqdm
from typing import List, Optional
import re
from .registry import ChunkerRegistry
from litellm import completion

@ChunkerRegistry.register("LLMSemanticChunker")
class LLMSemanticChunker(BaseChunker):
    def __init__(
        self, 
        litellm_config: Optional[dict] = None,
        length_type: str = 'token',
        **kwargs
    ):
        super().__init__(length_type=length_type, **kwargs)
        
        self._litellm_config = litellm_config or {}
        
        # Initialize the base splitter for initial text splitting
        self.splitter = RecursiveTokenChunker(
            chunk_size=50,
            chunk_overlap=0,
            length_function=self.length_function
        )

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        """Generate the prompt for the LLM."""
        base_prompt = (
            "You are an expert document analyzer. Split this text into thematic sections. "
            "Chunks are marked with <|start_chunk_X|> tags. Respond with 'split_after: ' "
            "followed by comma-separated chunk numbers where splits should occur."
        )

        user_content = (
            f"CHUNKED_TEXT: {chunked_input}\n\n"
            f"Respond with split points (ascending, ≥{current_chunk}). "
            "Include at least one split."
        )
        if invalid_response:
            user_content += (
                f"\n\\Previous invalid response: {invalid_response}. "
                "Do not repeat these numbers."
            )

        return [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_content}
        ]

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _get_llm_response(self, context: str, current: int) -> str:
        """Get chunking suggestions from LLM using LiteLLM."""
        try:
            response = completion(
                model=self._litellm_config.get('model', 'openai/gpt-4o'),
                messages=self.get_prompt(context, current),
                temperature=0.2,
                max_tokens=200,
                api_base=self._litellm_config.get('model_api_base')
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {str(e)}")
            return ""

    def _parse_response(self, response: str) -> List[int]:
        """Parse the LLM response to extract split points."""
        numbers = []
        if 'split_after:' in response:
            # Grab all numbers after the substring 'split_after:'
            numbers = list(map(int, re.findall(r'\d+', response.split('split_after:')[1])))
        return sorted(n for n in numbers if n >= 0)

    def _merge_chunks(self, chunks: List[str], indices: List[int]) -> List[str]:
        """Merge chunks based on split indices."""
        merged = []
        current = []
        for i, chunk in enumerate(chunks):
            current.append(chunk)
            if i in indices:
                merged.append(" ".join(current).strip())
                current = []
        if current:
            merged.append(" ".join(current).strip())
        return merged

    def split_text(self, text: str) -> List[str]:
        """Split input text into coherent chunks using LLM guidance."""
        chunks = self.splitter.split_text(text)
        split_indices = []
        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            while current_chunk < len(chunks) - 4:
                context_window = []
                token_count = 0

                for i in range(current_chunk, len(chunks)):
                    token_count += self.length_function(chunks[i])
                    if token_count > 800:
                        break
                    context_window.append(f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>")

                response = self._get_llm_response("\n".join(context_window), current_chunk)
                numbers = self._parse_response(response)

                if numbers:
                    split_indices.extend(numbers)
                    current_chunk = numbers[-1]
                    pbar.update(current_chunk - pbar.n)
                else:
                    break

        return self._merge_chunks(chunks, split_indices)