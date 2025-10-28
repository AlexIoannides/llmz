"""Tests for model training loops."""

import logging
import math
import re
from unittest.mock import Mock

import pytest
import torch
from _pytest.logging import LogCaptureFixture
from torch import nn
from torch.utils.data import DataLoader

from llmz.evaluate import Evaluator
from llmz.train import (
    GradientClipCallback,
    LinearWarmupCosineAnnealingLRSchedule,
    autoregressive_llm_loss,
    train,
)


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
