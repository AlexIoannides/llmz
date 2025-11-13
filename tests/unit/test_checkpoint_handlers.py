"""Tests for checkpoint handlers."""

import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from llmz.checkpoint_handlers import (
    STATE_DICT_FILE_EXT,
    LocalFSCheckpointHandler,
)

TrainingLoopObjects = tuple[
    nn.Module, optim.Optimizer, lr_scheduler.LRScheduler, dict[str, Any]
]


@pytest.fixture
def model_optim_lrs_meta() -> TrainingLoopObjects:
    model = nn.Linear(5, 10)
    optimiser = optim.SGD(model.parameters())
    lr_schedule = lr_scheduler.ConstantLR(optimiser, factor=1.0)
    return model, optimiser, lr_schedule, {"foo": "bar", "x": 1}


def test_LocalFSCheckpointHandler_creates_ckpt_dir(tmp_path: Path):
    ckpt_base_name = "llmz"
    with patch("llmz.checkpoint_handlers.LOCAL_FS_PATH", tmp_path):
        LocalFSCheckpointHandler(ckpt_base_name)
    assert (tmp_path / ckpt_base_name).exists()


def test_LocalFSCheckpointHandler_saves_checkpoints(
    tmp_path: Path, model_optim_lrs_meta: TrainingLoopObjects
):
    ckpt_base_name = "llmz"
    with patch("llmz.checkpoint_handlers.LOCAL_FS_PATH", tmp_path):
        checkpointer = LocalFSCheckpointHandler(ckpt_base_name)
    model, optimiser, lr_schedule, metadata = model_optim_lrs_meta
    checkpointer.save_checkpoint(model, optimiser, lr_schedule, 1, metadata)
    checkpointer.save_checkpoint(model, None, None, 2, {"A": "B", **metadata})

    files = list((tmp_path / ckpt_base_name).glob(f"*.{STATE_DICT_FILE_EXT}"))
    assert len(files) == 2

    state_dict_0 = torch.load(files[0], weights_only=False)
    assert state_dict_0["model"]["weight"].size() == (10, 5)
    assert state_dict_0["optimiser"]["param_groups"][0]["lr"] == 0.001
    assert state_dict_0["lr_schedule"]["factor"] == 1.0
    assert state_dict_0["step"] == 1
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", state_dict_0["timestamp"])
    assert state_dict_0["metadata"]["foo"] == "bar"

    state_dict_1 = torch.load(files[1], weights_only=False)
    assert state_dict_1["model"]["weight"].size() == (10, 5)
    assert state_dict_1["optimiser"] is None
    assert state_dict_1["lr_schedule"] is None
    assert state_dict_1["step"] == 2
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", state_dict_1["timestamp"])
    assert state_dict_1["metadata"]["A"] == "B"


def test_LocalFSCheckpointHandler_saves_checkpoints_raises_on_prohibited_overwrite(
    tmp_path: Path, model_optim_lrs_meta: TrainingLoopObjects
):
    ckpt_base_name = "llmz"
    with patch("llmz.checkpoint_handlers.LOCAL_FS_PATH", tmp_path):
        checkpointer = LocalFSCheckpointHandler(ckpt_base_name)
    model, optimiser, lr_schedule, metadata = model_optim_lrs_meta
    checkpointer.save_checkpoint(model, optimiser, lr_schedule, 1, metadata)
    expected_msg = "already exists and overwrite_existing=False"
    with pytest.raises(RuntimeError, match=expected_msg):
        checkpointer.save_checkpoint(model, optimiser, lr_schedule, 1, metadata)


def test_LocalFSCheckpointHandler_loads_checkpoints(
    tmp_path: Path, model_optim_lrs_meta: TrainingLoopObjects
):
    ckpt_base_name = "llmz"
    ckpt_dir = tmp_path / ckpt_base_name
    ckpt_dir.mkdir(exist_ok=True)

    model, optimiser, lr_schedule, metadata = model_optim_lrs_meta
    state_dict = {
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "lr_schedule": lr_schedule.state_dict(),
        "metadata": metadata,
    }

    state_dict["step"] = 1000
    state_dict["timestamp"] = "foo"
    ckpt_file_1 = ckpt_dir / f"1000.{STATE_DICT_FILE_EXT}"
    torch.save(state_dict, ckpt_file_1)

    state_dict["step"] = 2000
    state_dict["timestamp"] = "bar"
    ckpt_file_2 = ckpt_dir / f"2000.{STATE_DICT_FILE_EXT}"
    torch.save(state_dict, ckpt_file_2)

    # change model and optimiser state after persistence
    torch.nn.init.zeros_(model.weight)
    optimiser.param_groups[0]["lr"] = 0.0
    lr_schedule._step_count = 0

    with patch("llmz.checkpoint_handlers.LOCAL_FS_PATH", tmp_path):
        checkpointer = LocalFSCheckpointHandler(ckpt_base_name)

    ckpt = checkpointer.load_checkpoint(model, optimiser, lr_schedule, 1000)
    assert ckpt.model.weight.sum() != 0.0
    assert ckpt.optimiser is not None and ckpt.optimiser.param_groups[0]["lr"] != 0
    assert ckpt.lr_schedule is not None and ckpt.lr_schedule._step_count != 0
    assert ckpt.step == 1000

    ckpt = checkpointer.load_checkpoint(model, optimiser, lr_schedule, 2000)
    assert ckpt.model.weight.sum() != 0.0
    assert ckpt.optimiser is not None and ckpt.optimiser.param_groups[0]["lr"] != 0
    assert ckpt.lr_schedule is not None and ckpt.lr_schedule._step_count != 0
    assert ckpt.step == 2000

    ckpt = checkpointer.load_checkpoint(model, optimiser, lr_schedule, None)
    assert ckpt.step == 2000


def test_LocalFSCheckpointHandler_loads_checkpoints_raises_error_on_missing_state_dicts(
    tmp_path: Path, model_optim_lrs_meta: TrainingLoopObjects
):
    ckpt_base_name = "llmz"
    ckpt_dir = tmp_path / ckpt_base_name
    ckpt_dir.mkdir(exist_ok=True)

    model, optimiser, lr_schedule, metadata = model_optim_lrs_meta

    state_dict = {
        "model": model.state_dict(),
        "optimiser": None,
        "lr_schedule": None,
        "step": 1000,
        "timestamp": "foo",
        "metadata": metadata,
    }
    ckpt_file = ckpt_dir / f"1000.{STATE_DICT_FILE_EXT}"
    torch.save(state_dict, ckpt_file)

    with patch("llmz.checkpoint_handlers.LOCAL_FS_PATH", tmp_path):
        checkpointer = LocalFSCheckpointHandler(ckpt_base_name)

    with pytest.raises(RuntimeError, match="no optimiser in checkpoint"):
        checkpointer.load_checkpoint(model, optimiser, None)

    with pytest.raises(RuntimeError, match="no LR schedule in checkpoint"):
        checkpointer.load_checkpoint(model, None, lr_schedule)


def test_LocalFSCheckpointHandler_loads_checkpoints_raises_error_on_missing_files(
    tmp_path: Path, model_optim_lrs_meta: TrainingLoopObjects
):
    ckpt_base_name = "llmz"
    ckpt_dir = tmp_path / ckpt_base_name
    ckpt_dir.mkdir(exist_ok=True)

    model, optimiser, lr_schedule, _ = model_optim_lrs_meta
    with patch("llmz.checkpoint_handlers.LOCAL_FS_PATH", tmp_path):
        checkpointer = LocalFSCheckpointHandler(ckpt_base_name)

    with (
        patch.object(checkpointer, "list_checkpoints", return_value=[]),
        pytest.raises(FileExistsError, match="cannot find checkpoint at"),
    ):
        checkpointer.load_checkpoint(model, optimiser, None)

    with pytest.raises(FileExistsError, match="cannot find checkpoint at"):
        checkpointer.load_checkpoint(model, optimiser, lr_schedule, 1000)


def test_LocalFSCheckpointHandler_lists_checkpoints(
    tmp_path: Path, model_optim_lrs_meta: TrainingLoopObjects
):
    ckpt_base_name = "llmz"
    ckpt_dir = tmp_path / ckpt_base_name
    ckpt_dir.mkdir(exist_ok=True)

    model, optimiser, lr_schedule, metadata = model_optim_lrs_meta
    state_dict = {
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "metadata": metadata,
    }

    state_dict["step"] = 2000
    state_dict["timestamp"] = "bar"
    ckpt_file_2 = ckpt_dir / f"2000.{STATE_DICT_FILE_EXT}"
    torch.save(state_dict, ckpt_file_2)

    state_dict["step"] = 1000
    state_dict["timestamp"] = "foo"
    ckpt_file_1 = ckpt_dir / f"1000.{STATE_DICT_FILE_EXT}"
    torch.save(state_dict, ckpt_file_1)

    with patch("llmz.checkpoint_handlers.LOCAL_FS_PATH", tmp_path):
        checkpointer = LocalFSCheckpointHandler(ckpt_base_name)

    assert checkpointer.list_checkpoints() == [
        f"1000.{STATE_DICT_FILE_EXT}",
        f"2000.{STATE_DICT_FILE_EXT}",
    ]
