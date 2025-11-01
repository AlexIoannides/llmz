"""Pytest fixture and other testing common testing utilities."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ToyData(Dataset):
    """Simple dataset to use for testing."""

    def __init__(self, vocab_size: int = 10, max_length: int = 32, n_obs: int = 10):
        """Initialise."""
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.n_obs = n_obs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fake_tokens_chunk = torch.randint(
            0, self.vocab_size - 1, (self.max_length + 1,)
        )
        return fake_tokens_chunk[: self.max_length], fake_tokens_chunk[1:]

    def __len__(self) -> int:
        return self.n_obs


class ToyModel(nn.Module):
    """Simple model to use for testing."""

    def __init__(self, vocab_size: int = 10, embed_dim: int = 32):
        """Initialise."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ffn = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.embed(x)
        out = self.ffn(out)
        return out


@pytest.fixture
def dataset() -> Dataset:
    """Make dataset for testing."""
    return ToyData()


@pytest.fixture
def dataloader(dataset: Dataset) -> DataLoader:
    """Make dataloader for testing."""
    return DataLoader(dataset, 2)


@pytest.fixture
def model() -> nn.Module:
    """Make model for testing."""
    return ToyModel()


@pytest.fixture
def eval_metrics_fn() -> Callable[[nn.Module, DataLoader], dict[str, float]]:
    """Make evaluation metrics callable."""

    def f(m: nn.Module, dl: DataLoader) -> dict[str, float]:
        return {"loss": 0.1}

    return f


@pytest.fixture
def eval_scenarios_fn() -> Callable[[nn.Module], dict[str, str]]:
    """Make evaluation scenarios callable."""

    def f(m: nn.Module) -> dict[str, str]:
        return {"sample_text": "I've seen things..."}

    return f


@pytest.fixture
def text_data_file() -> Path:
    """Path to text file to use for training."""
    return Path("tests/resources/the-verdict.txt")
