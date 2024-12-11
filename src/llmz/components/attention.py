"""Attention blocks for transformer models."""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Basic attention block."""

    def __init__(
            self,
            context_size: int,
            dim_in: int,
            dim_out: int,
            n_heads: int = 1,
            dropout: float = 0.6,
            qkv_bias: bool = False
        ):
        """Initialise model.

        Args:
            dim_in: Dimension of input word embeddings.
            dim_out: Dimension of output attention embeddings.
            context_size: The number of input word embeddings in teh sequence.
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the model's forward pass."""
        return x


class ModelConfigError(Exception):
    """Custom exception class for model configuration errors."""

    pass
