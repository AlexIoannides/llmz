"""Tests for model training loops."""

from llmz.gpt2 import GPT2
from llmz.train import train


def test_train():
    model = GPT2(1, 1, 1)
    train(model)
    assert True
