"""Tokenizers for LLMs."""

from torch.utils.data import Dataset


class GPTSmallTextDataset(Dataset):
    """GPT dataset interface for any 'small' text data."""

    def __init__(self, text: str, max_length: int, *, stride: int = 1):
        """Initialise.

        Args:
            text: TODO
            max_length: TODO
            stride: TODO

        """
        pass
