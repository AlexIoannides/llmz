"""Tests for transformer blocks."""

import pytest
import torch
from torch import nn

from llmz.components.transformers import TransformerBlockGPT2


@pytest.mark.parametrize(
    "batch_size, context_size, dim_in", [(1, 2, 9), (1, 2, 9), (2, 4, 9)]
)
def test_TransformerBlock_output_properties(
    batch_size: int, context_size: int, dim_in: int
):
    tokens_batch = torch.ones(batch_size, context_size, dtype=torch.int32)
    embeddings_batch = nn.Embedding(10, dim_in)(tokens_batch)

    transformer = TransformerBlockGPT2(context_size, dim_in)
    out_batch = transformer(embeddings_batch)

    assert out_batch.size() == torch.Size((batch_size, context_size, dim_in))
    assert torch.all(torch.isreal(out_batch))
