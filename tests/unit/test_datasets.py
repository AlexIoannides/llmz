"""Test for datasets."""

import pytest
import tiktoken
import torch

from llmz.datasets import GPT2SmallTextDataset


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_GPT2SmallTextDataset_encodes_data(stride: int):
    text = "Attacks ships off the shoulder of Orion."
    max_len = 3

    tokenizer = tiktoken.get_encoding("gpt2")
    exp_tokens = tokenizer.encode(text)

    dataset = GPT2SmallTextDataset(text, max_len, stride)
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


def test_GPT2SmallTextDataset_raises_errors():
    text = "Attacks ships off the shoulder of Orion."
    max_len = 9

    with pytest.raises(RuntimeError, match="max_length"):
        GPT2SmallTextDataset(text, max_len)


def test_GPT2SmallTextDataset_create_data_loader():
    text = "Attacks ships off the shoulder of Orion."

    dataset = GPT2SmallTextDataset(text, max_length=3, stride=1)
    dataloader_iter = iter(dataset.create_data_loader(batch_size=2))
    b0_X, b0_y = next(dataloader_iter)
    b1_X, b1_y = next(dataloader_iter)

    exp_size = torch.Size([2, 3])
    assert b0_X.size() == b1_X.size() == b0_y.size() == b1_y.size() == exp_size
