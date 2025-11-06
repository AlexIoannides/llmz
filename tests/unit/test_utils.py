"""Tests for utility functions."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn, optim

from llmz.utils import STATE_DICT_FILE_EXT, LocalFSCheckpointHandler


@pytest.fixture
def model_optim_meta() -> tuple[nn.Module, optim.Optimizer, dict[str, Any]]:
    model = nn.Linear(5, 10)
    optimiser = optim.SGD(model.parameters())
    return model, optimiser, {"foo": "bar", "x": 1}


def test_LocalFSCheckpointHandler_creates_ckpt_dir(tmp_path: Path):
    ckpt_base_name = "llmz"
    with patch("llmz.utils.LOCAL_FS_PATH", tmp_path):
        LocalFSCheckpointHandler(ckpt_base_name)
    assert (tmp_path / ckpt_base_name).exists()


def test_LocalFSCheckpointHandler_saves_checkpoints(
    tmp_path: Path, model_optim_meta: tuple[nn.Module, optim.Optimizer, dict[str, Any]]
):
    ckpt_base_name = "llmz"
    with patch("llmz.utils.LOCAL_FS_PATH", tmp_path):
        checkpointer = LocalFSCheckpointHandler(ckpt_base_name)
    model, optimiser, metadata = model_optim_meta
    checkpointer.save_checkpoint(model, optimiser, 1, metadata)
    checkpointer.save_checkpoint(model, None, 2, {"A": "B", **metadata})

    files = list((tmp_path / ckpt_base_name).glob(f"*.{STATE_DICT_FILE_EXT}"))
    assert len(files) == 2

    state_dict_0 = torch.load(files[0], weights_only=False)
    assert state_dict_0["model"]["weight"].size() == (10, 5)
    assert state_dict_0["optimiser"]["param_groups"][0]["lr"] == 0.001
    assert state_dict_0["metadata"]["foo"] == "bar"

    state_dict_1 = torch.load(files[1], weights_only=False)
    assert state_dict_1["model"]["weight"].size() == (10, 5)
    assert state_dict_1["optimiser"] is None
    assert state_dict_1["metadata"]["A"] == "B"


def test_LocalFSCheckpointHandler_saves_checkpoints_raises_on_prohibited_overwrite(
    tmp_path: Path, model_optim_meta: tuple[nn.Module, optim.Optimizer, dict[str, Any]]
):
    ckpt_base_name = "llmz"
    with patch("llmz.utils.LOCAL_FS_PATH", tmp_path):
        checkpointer = LocalFSCheckpointHandler(ckpt_base_name)
    model, optimiser, metadata = model_optim_meta
    checkpointer.save_checkpoint(model, optimiser, 1, metadata)
    expected_msg = "already exists and overwrite_existing=False"
    with pytest.raises(RuntimeError, match=expected_msg):
        checkpointer.save_checkpoint(model, optimiser, 1, metadata)
