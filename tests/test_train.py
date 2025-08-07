"""Tests for model training loops."""

from llmz.gpt2 import GPT2
from llmz.train import CosineAnnealingLRSchedule, train


def test_CosineAnnealingLRSchedule():
    lr_schedule = CosineAnnealingLRSchedule()
    assert lr_schedule is not None


def test_train():
    model = GPT2(1, 1, 1)
    train(model)
    assert True
