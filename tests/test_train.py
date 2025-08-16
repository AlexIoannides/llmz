"""Tests for model training loops."""

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, Dataset

from llmz.gpt2 import GPT2
from llmz.train import Evaluator, LinearWarmupCosineAnnealingLRSchedule, train


class TestData(Dataset):
    """Simple dataset to use for testing."""

    def __init__(self, max_length: int = 256, n_obs: int = 10):
        """Initialise.

        Args:
            max_length: Number of tokens for each data instance. Defaults to 256.
            n_obs: Number of observations. Default to 10.

        """
        self.max_length = max_length
        self.n_obs = n_obs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fake_tokens_chunk = torch.randint(0, 10000, (self.max_length+1,))
        return fake_tokens_chunk[:self.max_length], fake_tokens_chunk[1:]

    def __len__(self) -> int:
        return self.n_obs

class TestModel(nn.Module):
    """Simple model to use for testing."""

    pass


@pytest.fixture
def dataset() -> Dataset:
    """Make dataset for testing."""
    return TestData()


@pytest.fixture
def dataloader(dataset: Dataset) -> DataLoader:
    """Make dataloader for testing."""
    return DataLoader(dataset, 2)


def test_LinearWarmupCosineAnnealingLRSchedule_input_validation():
    pattern = (
        "Invalid arguments for LR schedule\n"
        " num_steps <= 0\n"
        " warmup_steps > num_steps\n"
        " initial_lr <= 0.0\n"
        " peak_lr < initial_lr"
    )
    with pytest.raises(ValueError, match=pattern):
        LinearWarmupCosineAnnealingLRSchedule(
            num_steps=-1,
            warmup_steps=0,
            initial_lr=0.0,
            peak_lr=-1.0
        )


def test_LinearWarmupCosineAnnealingLRSchedule():
    lr_schedule = LinearWarmupCosineAnnealingLRSchedule(
        num_steps=100,
        warmup_steps=30,
        initial_lr=0.001,
        peak_lr=0.01
    )
    # At step 0, should be initial_lr
    assert lr_schedule(0) == 0.001
    # At warmup_steps, should be peak_lr
    assert lr_schedule(30) == pytest.approx(0.01, rel=1e-6)
    # At last step, should be initial_lr again (cosine annealing)
    assert lr_schedule(100) == pytest.approx(0.001, rel=1e-6)
    # Beyond last step, should remain ar initial_lr
    assert lr_schedule(200) == 0.001

    pattern = "step=-1, must be > 0"
    with pytest.raises(ValueError, match=pattern):
        lr_schedule(-1)


def test_evaluator_compute_metrics(dataloader: DataLoader):
    pass


def test_evaluator_compute_scenarios(dataloader: DataLoader):
    pass


def test_evaluator_computes_evaluations(dataloader: DataLoader):
    pass


def test_train():
    pass
