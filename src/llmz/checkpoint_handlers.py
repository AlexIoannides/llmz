"""Checkpoint handlers."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn, optim

LOCAL_FS_PATH = Path.cwd() / ".llmz_ckpts"
STATE_DICT_FILE_EXT = "pt"


class Checkpoint(NamedTuple):
    """Container for checkpoints."""

    model: nn.Module
    optimiser: optim.Optimizer | None
    step: int
    timestamp: str
    metadata: dict[str, Any]


class _CheckpointHandler(ABC):
    """Abstract interface for all checkpointing types."""

    @abstractmethod
    def __init__(self, ckpt_base_name: str, overwrite_existing: bool = False):
        """Initialise.

        Args:
            ckpt_base_name: Base name to give all checkpoint files.
            overwrite_existing: Whether to overwrite existing checkpoints. Defaults to
                False.

        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        model: nn.Module,
        optimiser: optim.Optimizer | None,
        step: int,
        extra_metadata: dict[str, Any],
    ) -> None:
        """Save checkpoint to chosen location.

        Args:
            model: The model with state dict to be persisted.
            optimiser: The optimiser with state dict to be persisted (optional).
            step: Training step that produced the model and optimiser.
            extra_metadata: Dictionary of additional related information to be persisted
                with model and optimiser.

        """
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        model: nn.Module,
        optimiser: optim.Optimizer | None,
        step: int | None = None,
    ) -> Checkpoint:
        """Load checkpoint.

        Args:
            model: The model to load the model state dict into.
            optimiser: The optimiser to load the model state dict into.
            step: The step associated with the persisted checkpoint (optional). If none,
                then the most recent will be returned automatically. Defaults to None.

        Returns:
            Model, optimiser (optional), and metadata checkpoint.

        """
        pass

    @abstractmethod
    def list_checkpoints(self) -> list[str]:
        """Get list of all checkpoints with base name.

        Returns:
            List of all checkpoint associated with the base name.

        """
        pass


class LocalFSCheckpointHandler(_CheckpointHandler):
    """Implementation of the Checkpointer interface for local FS persistence."""

    def __init__(
        self,
        ckpt_base_name: str,
        overwrite_existing: bool = False,
    ):
        """Initialise.

        Args:
            ckpt_base_name: Base name to give all checkpoint files.
            overwrite_existing: Whether to overwrite existing checkpoints. Defaults to
                False.

        """
        self._ckpts_dir = LOCAL_FS_PATH / ckpt_base_name
        self._ckpts_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite_existing = overwrite_existing

    def save_checkpoint(
        self,
        model: nn.Module,
        optimiser: optim.Optimizer | None,
        step: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint to chosen location.

        Args:
            model: The model with state dict to be persisted.
            optimiser: The optimiser with state dict to be persisted (optional).
            step: Training step that produced the model and optimiser.
            extra_metadata: Dictionary of additional related information to be persisted
                with model and optimiser. Defaults to None.

        Raises:
            RuntimeError if the checkpoint exists and `overwrite_existing` has been set
                to `False`

        """
        state_dict = {
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict() if optimiser else None,
            "step": step,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metadata": extra_metadata,
        }
        ckpt_path = self._ckpts_dir / f"{step}.{STATE_DICT_FILE_EXT}"
        if ckpt_path.exists() and not self.overwrite_existing:
            msg = f"{ckpt_path} already exists and overwrite_existing=False"
            raise RuntimeError(msg)
        else:
            torch.save(state_dict, ckpt_path)

    def load_checkpoint(
        self,
        model: nn.Module,
        optimiser: optim.Optimizer | None,
        step: int | None = None,
    ) -> Checkpoint:
        """Load checkpoint.

        Args:
            model: The model to load the model state dict into.
            optimiser: The optimiser to load the model state dict into.
            step: The step associated with the persisted checkpoint (optional). If None,
                then the most recent will be returned automatically. Defaults to None.

        Returns:
            Model, optimiser (optional), and metadata checkpoint.

        Raises:
            FileExistsError if checkpoint file cannot be located on local FS.

        """
        if step:
            ckpt_path = self._ckpts_dir / f"{step}.{STATE_DICT_FILE_EXT}"
        elif ckpts := self.list_checkpoints():
            ckpt_path = self._ckpts_dir / ckpts[-1]
        else:
            raise FileExistsError(f"cannot find checkpoint at {self._ckpts_dir}")

        if not ckpt_path.exists():
            raise FileExistsError(f"cannot find checkpoint at {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict["model"], strict=True)
        if optimiser:
            optimiser.load_state_dict(state_dict["optimiser"])
        return Checkpoint(
            model,
            optimiser,
            state_dict["step"],
            state_dict["timestamp"],
            state_dict["metadata"],
        )

    def list_checkpoints(self) -> list[str]:
        """Get list of all checkpoints with base name.

        Returns:
            List of all checkpoint associated with the base name in ascending order of
                steps.

        """
        ckpts = [
            str(ckpt.name) for ckpt in self._ckpts_dir.glob(f"*.{STATE_DICT_FILE_EXT}")
        ]
        return sorted(ckpts, key=lambda e: int(e.split(".")[0]))
