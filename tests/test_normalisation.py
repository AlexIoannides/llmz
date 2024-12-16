"""Test for normalisation modules."""

import torch

from llmz.components.normalisation import LayerNormalisation


def test_LayerNormalisation():
    torch.manual_seed(42)
    test_inputs_size = (3, 10)
    inputs = torch.randn(test_inputs_size)

    layer_norm = LayerNormalisation()
    inputs_norm = layer_norm(inputs)

    mean_inputs_norm = inputs_norm.mean(dim=-1, keepdim=True)
    var_inputs_norm = inputs_norm.var(dim=-1, keepdim=True)

    exp_mean_inputs_norm = torch.zeros(test_inputs_size)
    exp_var_inputs_norm = torch.ones(test_inputs_size)

    torch.testing.assert_close(mean_inputs_norm, exp_mean_inputs_norm)
    torch.testing.assert_close(var_inputs_norm, exp_var_inputs_norm)
