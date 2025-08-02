"""Attention blocks for transformer models."""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Basic causal attention block."""

    def __init__(
        self,
        context_size: int,
        dim_in: int,
        dim_out: int,
        n_heads: int = 1,
        dropout: float = 0.6,
        qkv_bias: bool = False,
    ):
        """Initialise module.

        Args:
            dim_in: Dimension of input word embeddings.
            dim_out: Dimension of output attention embeddings.
            context_size: The number of input word embeddings in the sequence.
            n_heads: The number of attention heads. Defaults to 1.
            dropout: The dropout rate. Defaults to 0.6.
            qkv_bias: Whether or not to include bias in the linear layers used to
                compute W_query, W_key and W_value. Defaults to False.

        Raises:
            ModelConfigError: if dim_out % n_heads

        """
        super().__init__()
        if dim_out % n_heads != 0:
            raise ModelConfigError("dim_out % n_heads != 0")

        self.dim_out = dim_out
        self.n_heads = n_heads
        self.dim_head = dim_out // n_heads  # // --> returns int
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_size, context_size), diagonal=1)
        )  # these are not parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the module's forward pass.

        Args:
            x: Batch of token embeddings.

        Returns:
            Batch of attention weighted embeddings.

        """
        batch_size, seq_len, dim_in = x.size()

        # get mask for sequence length
        mask_bool = self.mask.bool()[:seq_len, :seq_len]

        # single head (dim = batch_size, n_heads, seq_len, dim_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # split single head into multiple heads
        queries = queries.view(batch_size, seq_len, self.n_heads, self.dim_head)
        keys = keys.view(batch_size, seq_len, self.n_heads, self.dim_head)
        values = values.view(batch_size, seq_len, self.n_heads, self.dim_head)

        # reshape in size = batch_size, n_heads, seq_len, head_dim
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute attention scores (matrix multiplication works on final two dimensions)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # compute attention weights from attention scores (dim = -1 -> last dim in size)
        attn_weights = torch.softmax(attn_scores / keys.size()[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # compute context embeddings & reshape to batch_size, seq_len, n_heads, head_dim
        context_embeddings = (attn_weights @ values).transpose(1, 2)

        # reshape to factor-out the multiple heads and take into account dim_out
        context_embeddings = context_embeddings.view(batch_size, seq_len, self.dim_out)
        context_embeddings = self.out_proj(context_embeddings)
        return context_embeddings


class ModelConfigError(Exception):
    """Custom exception class for model configuration errors."""

    pass
