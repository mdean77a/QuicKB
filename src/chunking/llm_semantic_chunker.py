from .base_chunker import BaseChunker
# Fix EmbeddingManager import
from embeddings.base_embedder import EmbeddingManager
from .recursive_token_chunker import RecursiveTokenChunker
import anthropic
import backoff
from tqdm import tqdm
from typing import List
import re
from .registry import ChunkerRegistry 
@ChunkerRegistry.register("LLMSemanticChunker")
class LLMSemanticChunker(BaseChunker):
    def __init__(self, organisation: str = "openai", api_key: str = None, model_name: str = None):
        super().__init__()

        if organisation == "openai":
            from openai import OpenAI  # Possibly a custom wrapper
            self.client = OpenAI(api_key=api_key)
            self.model = model_name or "gpt-4o"
            self.organisation = "openai"
        elif organisation == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model_name or "claude-3-haiku-20240307"
            self.organisation = "anthropic"
        else:
            raise ValueError("Invalid organisation, choose 'openai' or 'anthropic'")

        self.splitter = RecursiveTokenChunker(
            chunk_size=50,
            chunk_overlap=0,
            length_function=EmbeddingManager.get_token_counter()
        )

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
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

    def split_text(self, text: str) -> List[str]:
        chunks = self.splitter.split_text(text)
        split_indices = []
        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            while current_chunk < len(chunks) - 4:
                context_window = []
                token_count = 0

                for i in range(current_chunk, len(chunks)):
                    token_count += EmbeddingManager.get_token_counter()(chunks[i])
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

    def _get_llm_response(self, context: str, current: int):
        @backoff.on_exception(backoff.expo, Exception, max_tries=3)
        def _send_request():
            if self.organisation == "openai":
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self.get_prompt(context, current),
                    temperature=0.2,
                    max_tokens=200
                ).choices[0].message.content
            else:
                return self.client.messages.create(
                    model=self.model,
                    messages=self.get_prompt(context, current),
                    max_tokens=200,
                    temperature=0.2
                ).content[0].text

        try:
            return _send_request()
        except Exception as e:
            print(f"LLM API error: {str(e)}")
            return ""

    def _parse_response(self, response: str):
        numbers = []
        if 'split_after:' in response:
            # Grab all numbers after the substring 'split_after:'
            numbers = list(map(int, re.findall(r'\d+', response.split('split_after:')[1])))
        return sorted(n for n in numbers if n >= 0)

    def _merge_chunks(self, chunks, indices):
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