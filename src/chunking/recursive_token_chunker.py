
# This script is adapted from the LangChain package, developed by LangChain AI.
# Original code can be found at: https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py
# chunking_evaluation modification: https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/recursive_token_chunker.py
# License: MIT License

from typing import Any, List, Optional
from .utils import Language
from .fixed_token_chunker import TextSplitter
import re
from .registry import ChunkerRegistry 

def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> List[str]:
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]

@ChunkerRegistry.register("RecursiveTokenChunker")
class RecursiveTokenChunker(TextSplitter):
    """Splitting text by recursively looking at characters / tokens."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        _good_splits = []
        actual_separator = "" if self._keep_separator else separator

        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, actual_separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, actual_separator)
            final_chunks.extend(merged_text)

        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        # Omitted for brevity – same as original
        # ... (the large if-elif chain of language separators) ...
        if language == Language.PYTHON:
            return [
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        # etc...
        raise ValueError(
            f"Language {language} is not supported! "
            f"Please choose from {list(Language)}"
        )