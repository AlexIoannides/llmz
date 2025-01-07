"""Test for GP2 model."""

import pytest
import torch

from llmz.gpt2 import GPT2


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
