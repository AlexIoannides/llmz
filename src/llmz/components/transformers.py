"""Transformer block for LLMs."""

import torch
from torch import nn

from llmz.components.activations import GELU
from llmz.components.attention import MultiHeadAttention
from llmz.components.normalisation import LayerNormalisation


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
        self.attention = MultiHeadAttention(
            context_size, dim_in, dim_out, n_heads, dropout, qkv_bias
        )
        self.linear_1 = nn.Linear(dim_out, dim_out * 2)
        self.linear_2 = nn.Linear(dim_out * 2, dim_out)
        self.normalise_1 = LayerNormalisation(dim_out)
        self.normalise_2 = LayerNormalisation(dim_out)
        self.dropout = nn.Dropout(dropout)
        self.gelu = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the module's forward pass.

        Args:
            x: Batch of token embeddings.

        Returns:
            Batch of attention weighted embeddings.

        """
        y1 = self.normalise_1(x)
        y1 = self.attention(y1)
        y1 = self.dropout(y1)

        y2 = self.normalise_2(y1 + x)
        y2 = self.linear_1(y2)
        y2 = self.gelu(y2)
        y2 = self.linear_2(y2)
        y2 = self.dropout(y2)

        return y1 + y2
