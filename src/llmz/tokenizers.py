"""Tokenizers for LLMs."""

import tiktoken
import torch
from torch.utils.data import Dataset


class GPTSmallTextDataset(Dataset):
    """GPT dataset interface for any 'small' text data.

    This will tokenize all text in-memory using a GPT2's tokenization algorithm, which
    is a pre-trained Bite Pair Encoding (BPE).
    """

    def __init__(self, text: str, max_length: int, stride: int = 1):
        """Initialise.

        Args:
            text: Raw text data to convert into tokens.
            max_length: Number of tokens for each data instance.
            stride: Separation (in tokens) between consecutive instances. Defaults to 1.

        """
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        n_instances = len(tokens) - max_length

        self._X = torch.ones((n_instances, max_length))
        self._y = torch.ones((n_instances, max_length))

        for i in range(0, n_instances, stride):
            self._X[i,] = torch.tensor(tokens[i : i + max_length])
            self._y[i,] = torch.tensor(tokens[i + 1 : i + max_length + 1])

    def __len__(self) -> int:
        return self._X.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx,], self._y[idx,]
