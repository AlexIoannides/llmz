"""Attention blocks for transformer models."""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Basic attention block."""

    def __init__(
            self,
            dim_in: int,
            dim_out: int,
            chunk_size: int,
            n_heads: int = 1,
            dropout: float = 0.6,
            qkv_bias: bool = False
        ):
        """Initialise model."""
        super().__init__()
        if dim_out % n_heads != 0:
            raise ModelConfigError("dim_out % n_heads != 0")

        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the model's forward pass."""
        return x


class ModelConfigError(Exception):
    """Custom exception class for model configuration errors."""

    pass
