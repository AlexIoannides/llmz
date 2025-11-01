"""Tools for tokenizing text.

Individual model definitions may have their own tokenizers defined alongside them,
but they will all implement the interface defined here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class _Tokenizer(ABC):
    """Abstract base class for text tokenizers."""

    def __call__(self, text: str) -> list[int]:
        return self.text2tokens(text)

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def text2tokens(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def tokens2text(self, token: list[int]) -> str:
        pass
