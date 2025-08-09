"""Tests for model training loops."""

import pytest

from llmz.gpt2 import GPT2
from llmz.train import CosineAnnealingWithWarmupLRSchedule, train


def test_CosineAnnealingLRSchedule_input_validation():
    pattern = (
        "Invlaid arguements for LR schedule\n"
        "num_steps <= 0\n"
        "warmup_steps < num_steps\n"
        "initial_lr <= 0.0\n"
        "peak_lr < initial_lr"
    )
    with pytest.raises(ValueError, match=pattern):
        CosineAnnealingWithWarmupLRSchedule(0, -1, 0.0, -1.0)


def test_CosineAnnealingLRSchedule():
    lr_schedule = CosineAnnealingWithWarmupLRSchedule(100, 25, 0.001, 0.01)
    assert lr_schedule is not None


def test_train():
    model = GPT2(1, 1, 1)
    train(model)
    assert True
