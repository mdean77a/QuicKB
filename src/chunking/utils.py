from enum import Enum
import os
import tiktoken
from typing import List, Optional, Callable
import openai
from openai import OpenAI

class EmbeddingManager:
    _custom_embedders = {}
    _custom_token_counters = {}
    
    @classmethod
    def register_embedder(cls, name: str, embedder: object):
        cls._custom_embedders[name] = embedder
        
    @classmethod
    def register_token_counter(cls, name: str, counter: Callable[[str], int]):
        cls._custom_token_counters[name] = counter

    @staticmethod
    def get_default_embedder() -> object:
        return OpenAIEmbedder()

    @staticmethod
    def get_default_token_counter() -> Callable[[str], int]:
        return openai_token_count

    @classmethod
    def get_embedder(cls, name: Optional[str] = None) -> object:
        if name and name in cls._custom_embedders:
            return cls._custom_embedders[name]
        return cls.get_default_embedder()

    @classmethod
    def get_token_counter(cls, name: Optional[str] = None) -> Callable[[str], int]:
        if name and name in cls._custom_token_counters:
            return cls._custom_token_counters[name]
        return cls.get_default_token_counter()
        

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
        return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

def openai_token_count(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string, disallowed_special=()))

class Language(str, Enum):
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