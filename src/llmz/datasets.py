"""Datasets for LLMs."""

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTSmallTextDataset(Dataset):
    """GPT dataset interface for any 'small' text data.

    This will tokenize all text in-memory using a GPT2's tokenization algorithm, which
    is a pre-trained Bite Pair Encoding (BPE).
    """

    def __init__(self, text: str, max_length: int = 256, stride: int = 128):
        """Initialise.

        Args:
            text: Raw text data to convert into tokens.
            max_length: Number of tokens for each data instance. Defaults to 256.
            stride: Separation (in tokens) between consecutive instances. Defaults to
                128.

        """
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)

        n_tokens = len(tokens)
        n_instances = int((n_tokens - max_length) / stride)
        if n_instances == 0:
            raise RuntimeError("max_length + stride <= number of tokens")

        self._X = torch.ones((n_instances, max_length))
        self._y = torch.ones((n_instances, max_length))

        for n, i in enumerate(range(0, n_instances*stride, stride)):
            self._X[n,] = torch.tensor(tokens[i : i + max_length])
            self._y[n,] = torch.tensor(tokens[i + 1 : i + max_length + 1])

        self.vocab_size = tokenizer.n_vocab

    def create_data_loader(
        self,
        batch_size: int = 4,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create data loader.

        Args:
            batch_size: The batch size. Defaults to 4.
            shuffle: Whether to randomise instance order after each iteration. Defaults
                to True.
            drop_last: Drop last batch if less than `batch_size`. Defaults to True.
            num_workers: Number of CPU processes to use for pre-processing. Defaults to
                0.

        Returns:
            A fully configured DataLoader

        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    def __len__(self) -> int:
        return self._X.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx,], self._y[idx,]
