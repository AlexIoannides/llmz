"""Tests for utility functions."""

from pathlib import Path
from unittest.mock import patch

from llmz.utils import LocalFSCheckpointHandler


def test_LocalFSCheckpointHandler_creates_ckpt_dir(tmp_path: Path):
    ckpt_base_name = "llmz"
    with patch("llmz.utils.LOCAL_FS_PATH") as local_ckpt_dir:
        local_ckpt_dir.return_value = tmp_path
        LocalFSCheckpointHandler(ckpt_base_name)
    assert (tmp_path / ckpt_base_name).exists()
