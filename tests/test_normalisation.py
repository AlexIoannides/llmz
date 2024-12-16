"""Test for normalisation modules."""

import torch

from llmz.components.normalisation import LayerNormalisation


def test_LayerNormalisation():
    torch.manual_seed(42)
    test_inputs_size = (3, 1000)
    inputs = 1.1 + 0.25 * torch.randn(test_inputs_size)

    layer_norm = LayerNormalisation(dim_in=test_inputs_size[1])
    inputs_norm = layer_norm(inputs)

    mean_inputs_norm = inputs_norm.mean(dim=-1, keepdim=True)
    var_inputs_norm = inputs_norm.var(dim=-1, keepdim=True)

    exp_mean_inputs_norm = torch.zeros((test_inputs_size[0], 1))
    exp_var_inputs_norm = torch.ones((test_inputs_size[0], 1))
    tolerance = {"atol": 0.001, "rtol": 0.001}

    torch.testing.assert_close(mean_inputs_norm, exp_mean_inputs_norm, **tolerance)
    torch.testing.assert_close(var_inputs_norm, exp_var_inputs_norm, **tolerance)
