"""Attention blocks for transformer models."""

import torch
from torch import nn


class Attention(nn.Module):
    """Basic attention block."""

    def __init__(
            self, dim_in: int, dim_out: int, chunk_size: int, qkv_bias: bool = False
        ):
        """Initialise model."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the model's forward pass."""
        return x
