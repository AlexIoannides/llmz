"""Test for normalisation modules."""

import pytest
import torch

from llmz.components.normalisation import LayerNormalisation


@pytest.mark.parametrize("location, scale", [(0.1, 0.025), (1.1, 0.25), (10.1, 2.5)])
def test_LayerNormalisation(location: float, scale: float):
    torch.manual_seed(42)
    test_inputs_size = (3, 1000)
    inputs = location + scale * torch.randn(test_inputs_size)

    layer_norm = LayerNormalisation(dim_in=test_inputs_size[1])
    inputs_norm = layer_norm(inputs)

    mean_inputs_norm = inputs_norm.mean(dim=-1, keepdim=True)
    var_inputs_norm = inputs_norm.var(dim=-1, keepdim=True)

    exp_mean_inputs_norm = torch.zeros((test_inputs_size[0], 1))
    exp_var_inputs_norm = torch.ones((test_inputs_size[0], 1))

    tol = {"atol": 0.001, "rtol": 0.001}

    torch.testing.assert_close(
        mean_inputs_norm, exp_mean_inputs_norm, rtol=tol["rtol"], atol=tol["atol"]
    )
    torch.testing.assert_close(
        var_inputs_norm, exp_var_inputs_norm, rtol=tol["rtol"], atol=tol["atol"]
    )
