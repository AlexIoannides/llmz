"""Tests for model training loops."""

import logging
import math
import re
from collections.abc import Callable
from unittest.mock import Mock

import pytest
import torch
from _pytest.logging import LogCaptureFixture
from torch import nn
from torch.utils.data import DataLoader, Dataset

from llmz.train import (
    Evaluator,
    GradientClipCallback,
    LinearWarmupCosineAnnealingLRSchedule,
    autoregressive_llm_loss,
    basic_llm_metrics,
    log,
    train,
)


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
    def f(m: nn.Module, dl: DataLoader) -> dict[str, float]:
        return {"loss": 0.1}

    return f


@pytest.fixture
def eval_scenarios_fn() -> Callable[[nn.Module], dict[str, str]]:
    def f(m: nn.Module) -> dict[str, str]:
        return {"sample_text": "I've seen things..."}

    return f


def test_LinearWarmupCosineAnnealingLRSchedule_input_validation():
    pattern = re.escape(
        "Invalid arguments for LR schedule\n"
        " * num_steps <= 0\n"
        " * warmup_steps > num_steps\n"
        " * initial_lr <= 0.0\n"
        " * peak_lr < initial_lr"
    )
    with pytest.raises(ValueError, match=pattern):
        LinearWarmupCosineAnnealingLRSchedule(
            num_steps=-1, warmup_steps=0, initial_lr=0.0, peak_lr=-1.0
        )


def test_LinearWarmupCosineAnnealingLRSchedule():
    lr_schedule = LinearWarmupCosineAnnealingLRSchedule(
        num_steps=100, warmup_steps=30, initial_lr=0.001, peak_lr=0.01
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


def test_evaluator_computes_evaluation_metrics(
    model: nn.Module, dataloader: DataLoader, eval_metrics_fn: Callable
):
    eval = Evaluator(dataloader, dataloader, eval_metrics_fn)
    eval.evaluate(1, model)
    eval.evaluate(2, model)
    assert len(eval._eval_records) == 2
    assert eval._eval_records[0].step == 1
    assert eval._eval_records[0].results == {"train_loss": 0.1, "val_loss": 0.1}


def test_evaluator_computes_evaluation_scenarios(
    model: nn.Module,
    dataloader: DataLoader,
    eval_metrics_fn: Callable,
    eval_scenarios_fn: Callable,
):
    eval = Evaluator(dataloader, dataloader, eval_metrics_fn, eval_scenarios_fn)
    eval.evaluate(1, model)
    eval.evaluate(2, model)
    assert eval._eval_records[0].results["sample_text"] == "I've seen things..."
    assert eval._eval_records[1].results["sample_text"] == "I've seen things..."


# fails - needs fixing and finishing
def test_evaluator_logs_evaluations(
    caplog: LogCaptureFixture, model: nn.Module, dataloader: DataLoader, eval_metrics_fn
):
    eval = Evaluator(dataloader, dataloader, eval_metrics_fn)
    with caplog.at_level(logging.INFO):
        eval.evaluate(1, model, log)
    assert "train_loss=0.1" in caplog.text
    assert "val_loss=0.1" in caplog.text


# TODO: implement test
def test_basic_llm_metrics():
    assert basic_llm_metrics is not None


def test_GradientClipCallback(model: nn.Module, dataloader: DataLoader):
    grad_clipper = GradientClipCallback(0.1)

    X, _ = next(iter(dataloader))
    out = model(X)
    out.flatten().mean().backward()

    max_grad_before_clip = max(
        p.grad.abs().max().item() for p in model.parameters() if p.grad is not None
    )

    grad_clipper(model)

    max_grad_after_clip = max(
        p.grad.abs().max().item() for p in model.parameters() if p.grad is not None
    )

    assert max_grad_before_clip > max_grad_after_clip


def test_train_runs_all_steps_end_to_end(
    model: nn.Module, dataloader: DataLoader, caplog: LogCaptureFixture
):
    mock_loss_calc = Mock(autoregressive_llm_loss)
    mock_loss_calc.side_effect = lambda m, X, _: m(X).flatten().mean()  # misc f(model)

    mock_optimiser = Mock(torch.optim.AdamW)
    mock_lr_schedule = Mock(torch.optim.lr_scheduler.LRScheduler)
    mock_evaluator = Mock(Evaluator)
    mock_callback = Mock(GradientClipCallback)

    epochs = 2
    steps_per_epoch = len(dataloader)
    eval_freq = 5
    log_freq = 2

    total_steps = epochs * steps_per_epoch

    assert all(p.grad is None for p in model.parameters())

    # with callbacks
    with caplog.at_level(logging.INFO):
        train(
            model=model,
            loss_calc=mock_loss_calc,
            optimiser=mock_optimiser,
            lr_schedule=mock_lr_schedule,
            train_dataloader=dataloader,
            train_epochs=epochs,
            model_backward_callbacks=[mock_callback, mock_callback],
            eval_freq_steps=eval_freq,
            evaluator=mock_evaluator,
            log_freq_steps=log_freq,
        )

    assert all(p.grad is not None for p in model.parameters())

    assert mock_optimiser.step.call_count == total_steps
    assert mock_optimiser.zero_grad.call_count == total_steps
    assert mock_lr_schedule.step.call_count == total_steps
    assert mock_evaluator.evaluate.call_count == total_steps // eval_freq
    assert mock_callback.call_count == 2 * total_steps

    assert len(caplog.records) == total_steps // log_freq
    assert caplog.messages[0] == "step=2, epoch=1"
    assert caplog.messages[-1] == "step=10, epoch=2"

    # without callbacks
    mock_callback.reset_mock()

    with caplog.at_level(logging.INFO):
        train(
            model=model,
            loss_calc=mock_loss_calc,
            optimiser=mock_optimiser,
            lr_schedule=mock_lr_schedule,
            train_dataloader=dataloader,
            train_epochs=epochs,
            model_backward_callbacks=None,
            eval_freq_steps=eval_freq,
            evaluator=mock_evaluator,
            log_freq_steps=log_freq,
        )

    assert mock_callback.call_count == 0


def test_autoregressive_llm_loss(model: nn.Module):
    X_batch = torch.zeros([2, 2, 3], dtype=torch.int)
    y_batch = torch.tensor([[0, 1], [2, 1]])

    mock_model = Mock(model)
    mock_model.return_value = torch.tensor(
        [
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.5], [0.0, 0.5, 0.0]],
        ]
    )

    # row-wise softmax of the patched model return values w/ flatten(0, 1)
    probs = torch.tensor(
        [
            [0.4519, 0.2741, 0.2741],
            [0.2741, 0.4519, 0.2741],
            [0.2741, 0.2741, 0.4519],
            [0.2741, 0.4519, 0.2741],
        ]
    )

    expected_loss = -0.25 * (
        1 * math.log(probs[0, 0])
        + 0 * math.log(probs[0, 1])
        + 0 * math.log(probs[0, 2])
        + 0 * math.log(probs[1, 0])
        + 1 * math.log(probs[1, 1])
        + 0 * math.log(probs[1, 2])
        + 0 * math.log(probs[2, 0])
        + 0 * math.log(probs[2, 1])
        + 1 * math.log(probs[2, 2])
        + 0 * math.log(probs[3, 0])
        + 1 * math.log(probs[3, 1])
        + 0 * math.log(probs[3, 2])
    )

    actual_loss = autoregressive_llm_loss(mock_model, X_batch, y_batch).item()

    assert actual_loss == pytest.approx(expected_loss, rel=0.001)
