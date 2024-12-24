"""Tests for transformer blocks."""

import pytest
import torch
from torch import nn

from llmz.components.transformers import TransformerBlock


@pytest.mark.parametrize("batch_size, context_size, dim_out", [(1, 2, 5), (2, 4, 3)])
def test_TransformerBlock_output_size(batch_size: int, context_size: int, dim_out: int):
    dim_in = 9

    tokens_batch = torch.ones(batch_size, context_size, dtype=torch.int32)
    embeddings_batch = nn.Embedding(10, dim_in)(tokens_batch)

    transformer = TransformerBlock(context_size, dim_in, dim_out)
    out_batch = transformer(embeddings_batch)

    assert out_batch.size() == torch.Size((batch_size, context_size, dim_out))
