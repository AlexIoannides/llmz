"""Activation functions for transformer models."""

import torch
from torch import nn


class GELU(nn.Module):
    """Guassian Error Linear Unit (GELU).

    Implemented using an approximation to `x * F(x)`, where `F` is the cumulative
    normal distribution function.
    """

    def __init__(self):
        """Initialise module."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the module's forward pass.

        Args:
            x: Batch of input tensors.

        Returns:
            Batch of output tensors that have been filtered on an element-by-element
                basis using the GELU activation function.

        """
        tanh_exponent = torch.sqrt(
            torch.tensor(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))
        )
        gelu_x = 0.5 * x * (1.0 + torch.tanh(tanh_exponent))
        return gelu_x
