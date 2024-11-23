"""Test for tokenizers."""

import pytest
import tiktoken

from llmz.datasets import GPTSmallTextDataset


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_GPTDataset_encodes_data(stride: int):
    text = "Attacks ships off the shoulder of Orion."
    max_len = 3

    tokenizer = tiktoken.get_encoding("gpt2")
    exp_tokens = tokenizer.encode(text)

    dataset = GPTSmallTextDataset(text, max_len, stride)
    X_0, y_0 = dataset[0]
    X_1, y_1 = dataset[1]

    assert len(dataset) == len(range(0, len(exp_tokens) - max_len, stride))
    assert len(X_0) == len(y_0) == max_len

    assert X_0[1] == y_0[0]
    assert X_1[1] == y_1[0]

    assert X_0[-1] == exp_tokens[max_len - 1]
    assert y_0[-1] == exp_tokens[max_len]

    assert X_1[-1] == exp_tokens[stride + max_len - 1]
    assert y_1[-1] == exp_tokens[stride + max_len]


def test_GPTDataset_raises_errors():
    text = "Attacks ships off the shoulder of Orion."
    max_len = 9

    with pytest.raises(RuntimeError, match="max_length"):
        GPTSmallTextDataset(text, max_len)
