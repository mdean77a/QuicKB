from enum import Enum
import os
import tiktoken
from typing import List, Optional
import openai
from openai import OpenAI

class OpenAIEmbedder:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-large",
        dimensions: Optional[int] = None,
    ):
        """
        Initialize the OpenAI Embedder.
        
        Args:
            api_key (str, optional): Your OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
            model_name (str, optional): Model to use for embeddings. Defaults to "text-embedding-3-large"
            dimensions (int, optional): Number of dimensions for the embeddings. Only for text-embedding-3 or later
        """
        self._api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self._api_key is None:
            raise ValueError(
                "Please provide an OpenAI API key or set the OPENAI_API_KEY environment variable"
            )
        
        self._client = OpenAI(api_key=self._api_key)
        self._model_name = model_name
        self._dimensions = dimensions

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts (List[str]): List of texts to get embeddings for
            
        Returns:
            List[List[float]]: List of embeddings vectors
        """
        # Replace newlines which can affect performance
        texts = [t.replace("\n", " ") for t in texts]
        
        response = self._client.embeddings.create(
            input=texts,
            model=self._model_name,
            dimensions=self._dimensions
        )
        
        # Sort embeddings by index to maintain input order
        sorted_embeddings = sorted(response.data, key=lambda e: e.index)
        return [result.embedding for result in sorted_embeddings]

def get_openai_embedding_function():
    """Returns an instance of OpenAIEmbedder with default settings."""
    return OpenAIEmbedder(model_name="text-embedding-3-large")

def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

class Language(str, Enum):
    """Enum of the programming languages."""
    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"