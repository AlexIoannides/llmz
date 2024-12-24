"""Normalisation operations."""

import torch
from torch import nn


class LayerNormalisation(nn.Module):
    """Layer normalisation.

    Normalises batches of input tensors close zero mean and unit variance. The module
    allows for some trained deviation from the a mean of zero and a variance of one.
    """

    def __init__(self, dim_in: int):
        """Initialise module.

        Args:
            dim_in: Dimension of the input batches.

        """
        super().__init__()
        self.epsilon = 1e-5

        # Trainable element-by-element adjustments to output tensors
        self.shift = nn.Parameter(torch.zeros(dim_in))
        self.scale = nn.Parameter(torch.ones(dim_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of module.

        Args:
            x: input tensors.

        Returns:
            Tensor-by-tensor normalised version of the inputs.

        """
        x_mean = x.mean(dim=-1, keepdim=True)
        x_stdev = x.std(dim=-1, keepdim=True, unbiased=False)  # unbiased as n -> inf
        x_norm = (x - x_mean) / (x_stdev + self.epsilon)
        breakpoint()
        return self.shift + self.scale * x_norm
