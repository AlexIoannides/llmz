"""Tests for custom activation functions."""

import pytest
import torch

from llmz.components.activations import GELU


@pytest.mark.parametrize(
    "x, result",
    [
        (torch.tensor(0.), torch.tensor(0.)),
        (torch.tensor([0., 0.]), torch.tensor([0., 0.]))
    ]
)
def test_GELU(x: torch.Tensor, result: torch.Tensor):
    gelu = GELU()
    torch.testing.assert_close(gelu(x), result, atol=0., rtol=0.)
