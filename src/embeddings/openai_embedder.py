import os
from typing import List, Optional
from openai import OpenAI

class OpenAIEmbedder:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-large",
        dimensions: Optional[int] = None,
    ):
        self._api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self._api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self._client = OpenAI(api_key=self._api_key)
        self._model_name = model_name
        self._dimensions = dimensions

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        response = self._client.embeddings.create(
            input=texts,
            model=self._model_name,
            dimensions=self._dimensions
        )
        # Sort by index so the embeddings line up with the original input order
        return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]