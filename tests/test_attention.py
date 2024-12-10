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

    attention = MultiHeadAttention(embedding_dim, attention_dim, context_size)
    out_batch = attention(embeddings_batch)

    assert out_batch.size() == torch.Size((batch_size, context_size, embedding_dim))


def test_Attention_raises_exception_on_bad_config():
    with pytest.raises(ModelConfigError, match="dim_out % n_heads != 0"):
        MultiHeadAttention(5, 5, 3, 3)
