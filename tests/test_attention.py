"""Tests for transformer."""

import torch
from torch import nn

from llmz.components.attention import MultiHeadAttention


def test_Attention_output_size():
    batch_size = 2
    chunk_size = 4
    embedding_dim = 5
    attention_dim = embedding_dim

    tokens_batch = torch.ones(batch_size, chunk_size, dtype=torch.int32)
    embeddings_batch = nn.Embedding(10, embedding_dim)(tokens_batch)

    attention = MultiHeadAttention(embedding_dim, attention_dim, chunk_size)
    out_batch = attention(embeddings_batch)

    assert out_batch.size() == torch.Size((batch_size, chunk_size, embedding_dim))
