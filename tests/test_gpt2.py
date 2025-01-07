"""Test for GP2 model."""

import re

import pytest
import torch

from llmz.gpt2 import GPT2, GPT2InferenceError


@pytest.mark.parametrize(
    "vocab_size, batch_size, context_size, embed_dim, n_tsfmr_blocks",
    [(10, 1, 2, 4, 1), (20, 1, 3, 8, 2), (27, 2, 4, 12, 3)],
)
def test_TransformerBlock_output_properties(
    vocab_size: int,
    batch_size: int,
    context_size: int,
    embed_dim: int,
    n_tsfmr_blocks: int,
):
    tokens_batch = torch.randint(
        0, vocab_size, (batch_size, context_size), dtype=torch.int32
    )

    model = GPT2(vocab_size, embed_dim, context_size, n_tsfmr_blocks)
    out_batch = model(tokens_batch)

    assert out_batch.size() == torch.Size((batch_size, context_size, vocab_size))
    assert torch.all(torch.isreal(out_batch))


def test_TransformerBlock_raises_error_for_incomputable_inference():
    model = GPT2(vocab_size=5, embed_dim=5, context_size=5, n_tsfmr_blocks=1)
    expected_err_msg = re.escape("seq_len (6) > context_size (5)")
    with pytest.raises(GPT2InferenceError, match=expected_err_msg):
        x = torch.ones((1, 6))
        model(x)
