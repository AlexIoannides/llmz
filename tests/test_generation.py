"""Tests for text generation tools."""


import pytest
import torch

from llmz.generate import _sample_decoding, _top_k_decoding


@pytest.mark.parametrize(
    "logits, temperature, token1_expected_pct",
    [(torch.tensor([1.0, 1.0]), 1.0, 0.5),
     (torch.tensor([1.0, 10.0]), 1.0, 1.0),
     (torch.tensor([1.0, 10.0]), 1000.0, 0.5),
     (torch.tensor([1.0, 2.0]), 1.0, 0.75),
     (torch.tensor([1.0, 2.0]), 0.001, 1.0)]
)
def test_sample_decoding(
    logits: torch.Tensor, temperature: float, token1_expected_pct: int
):
    torch.manual_seed(42)
    n_samples = 1000
    token1_sampled = torch.tensor(
        [_sample_decoding(logits, temperature) == 1 for _ in range(n_samples)]
    )
    token1_pct = token1_sampled.sum() / n_samples
    assert float(token1_pct) == pytest.approx(token1_expected_pct, rel=0.05)
