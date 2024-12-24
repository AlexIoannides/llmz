"""Transformer block for LLMs."""

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Basic transformer block with multi-head attention."""

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
            context_size: The number of input word embeddings in teh sequence.
            n_heads: The number of attention heads. Defaults to 1.
            dropout: The dropout rate. Defaults to 0.6.
            qkv_bias: Whether or not to include bias in the linear layers used to
                compute W_query, W_key and W_value. Defaults to False.

        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the module's forward pass.

        Args:
            x: Batch of token embeddings.

        Returns:
            Batch of attention weighted embeddings.

        """
        return x
