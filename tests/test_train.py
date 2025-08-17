"""Tests for model training loops."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from llmz.train import Evaluator, LinearWarmupCosineAnnealingLRSchedule, train


class TestData(Dataset):
    """Simple dataset to use for testing."""

    def __init__(self, vocab_size: int = 10, max_length: int = 32, n_obs: int = 10):
        """Initialise."""
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.n_obs = n_obs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fake_tokens_chunk = torch.randint(0, self.vocab_size-1, (self.max_length+1,))
        return fake_tokens_chunk[:self.max_length], fake_tokens_chunk[1:]

    def __len__(self) -> int:
        return self.n_obs


class TestModel(nn.Module):
    """Simple model to use for testing."""

    def __init__(self, vocab_size: int = 10, embed_dim: int = 32):
        """Initialise."""
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
    return TestData()


@pytest.fixture
def dataloader(dataset: Dataset) -> DataLoader:
    """Make dataloader for testing."""
    return DataLoader(dataset, 2)


@pytest.fixture
def model() -> nn.Module:
    """Make model for testing."""
    return TestModel()



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


def test_evaluator_compute_metrics(model: nn.Module, dataloader: DataLoader):
    metrics = Evaluator._compute_metrics(model, dataloader)
    assert metrics is not None


def test_evaluator_compute_scenarios(model: nn.Module):
    scenarios = Evaluator._compute_scenarios(model)
    assert scenarios is not None


def test_evaluator_computes_evaluations(model: nn.Module, dataloader: DataLoader):
    eval = Evaluator(dataloader, dataloader)
    assert eval is not None


def test_train_runs_all_steps_end_to_end():
    # calls loss_calc
    # calls evaluator
    # calls lr_schedule
    # calls clip_grads
    # updates model weights
    pass
