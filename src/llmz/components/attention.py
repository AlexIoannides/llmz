"""Attention blocks for transformer models."""

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Basic attention block."""

    def __init__(
            self, dim_in: int, dim_out: int, chunk_size: int, qkv_bias: bool = False
        ):
        """Initialise model."""
        super().__init__()
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the model's forward pass."""
        return x
