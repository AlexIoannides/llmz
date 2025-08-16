"""Tests for model training loops."""

import pytest

from llmz.gpt2 import GPT2
from llmz.train import LinearWarmupCosineAnnealingLRSchedule, train


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
    assert lr_schedule(30) == 0.01
    # At last step, should be initial_lr again (cosine annealing)
    assert lr_schedule(100) == pytest.approx(0.0, rel=1e-6)

    pattern = "step=101 not in the range [0, {self.num_steps}]"
    with pytest.raises(ValueError, match=pattern):
        lr_schedule(101)


def test_train():
    model = GPT2(1, 1, 1)
    train(model)
    assert True
