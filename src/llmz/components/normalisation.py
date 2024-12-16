"""Normalisation operations."""

import torch
from torch import nn


class LayerNormalisation(nn.Module):
    """Layer normalisation.

    Normalises batches of input tensor to zero mean and unit variance.
    """

    def __init(self):
        """Initialise module."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of module.

        Args:
            x: input tensors.

        Returns:
            Tensor-by-tensor normalised version of the inputs.

        """
        return x
