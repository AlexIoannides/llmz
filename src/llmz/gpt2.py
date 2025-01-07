"""Implementation of GPT2."""

import torch
from torch import nn

from llmz.components.normalisation import LayerNormalisation
from llmz.components.transformers import TransformerBlockGPT2


class GPT2(nn.Module):
    """Implementation of OpenAI's GPT2 model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_size: int,
        n_tsfmr_blocks: int = 1,
        n_attn_heads: int = 1,
        dropout: float = 0.6,
        qkv_bias: bool = False,
    ):
        """Initialise model.

        Args:
            vocab_size: The number of unique tokens that the model expects to encounter.
            embed_dim: Dimension of input word embeddings.
            context_size: The number of input word embeddings in the sequence.
            n_tsfmr_blocks: The number of transformer blocks stacked together.
            n_attn_heads: The number of attention heads in every transformer block.
                Defaults to 1.
            dropout: The dropout rate. Defaults to 0.6.
            qkv_bias: Whether or not to include bias in the linear layers used to
                compute W_query, W_key and W_value. Defaults to False.

        """
        super().__init__()

        self.context_size = context_size
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(context_size, embed_dim)
        self.dropout_embed = nn.Dropout(p=dropout)

        self.tsfmr_stack = nn.Sequential(
            *[
                TransformerBlockGPT2(
                    context_size, embed_dim, n_attn_heads, dropout, qkv_bias
                )
                for _ in range(n_tsfmr_blocks)
            ]
        )

        self.final_norm = LayerNormalisation(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the module's forward pass.

        Args:
            x: Batch of token embeddings.

        Returns:
            Batch of attention weighted embeddings.

        """
        _, seq_len = x.size()
        if seq_len > self.context_size:
            raise Exception("TODO")

        positions = torch.arange(0, seq_len, device=x.device)
        y = self.token_embed(x) + self.position_embed(positions)
        y = self.dropout_embed(y)
        y = self.tsfmr_stack(y)
        y = self.final_norm(y)
        logits = self.output_head(y)
        return logits
