"""Test for GP2 model."""

from llmz.gpt2 import GPT2


def test_GPT2_output_shape():
    model = GPT2()
    assert model is not None
