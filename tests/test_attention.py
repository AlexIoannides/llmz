"""Tests for transformer."""

import pytest
import torch
from torch import nn

from llmz.components.attention import ModelConfigError, MultiHeadAttention


def test_Attention_output_size():
    batch_size = 2
    context_size = 4
    embedding_dim = 5
    attention_dim = embedding_dim

    tokens_batch = torch.ones(batch_size, context_size, dtype=torch.int32)
    embeddings_batch = nn.Embedding(10, embedding_dim)(tokens_batch)

    attention = MultiHeadAttention(context_size, embedding_dim, attention_dim)
    out_batch = attention(embeddings_batch)

    assert out_batch.size() == torch.Size((batch_size, context_size, embedding_dim))


def test_Attention_head_dim():
    assert MultiHeadAttention(6, 6, dim_out=4, n_heads=1).dim_head == 4
    assert MultiHeadAttention(6, 6, dim_out=4, n_heads=2).dim_head == 2

    assert MultiHeadAttention(6, 6, dim_out=6, n_heads=1).dim_head == 6
    assert MultiHeadAttention(6, 6, dim_out=6, n_heads=2).dim_head == 3
    assert MultiHeadAttention(6, 6, dim_out=6, n_heads=3).dim_head == 2


def test_Attention_raises_exception_on_bad_config():
    with pytest.raises(ModelConfigError, match="dim_out % n_heads != 0"):
        MultiHeadAttention(3, 5, 5, 3)
