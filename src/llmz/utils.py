"""Utilities for working with LLMs."""


from pathlib import Path
from typing import Any, Literal, NamedTuple

import torch
from torch import nn

LOCAL_FS_PATH = Path.cwd() / ".llmz_ckpts"

CheckpointLocation = Literal["LocalFS"]


class Checkpoint(NamedTuple):
    """Container for checkpoints."""

    model: nn.Module
    optimiser: nn.Module | None
    metadata: dict[str, Any]


class CheckpointHandler:
    """Handle saving and loading checkpoint."""

    def __init__(
            self,
            ckpt_base_name: str,
            ckpt_location: CheckpointLocation = "LocalFS",
            overwrite_existing: bool = False
        ):
        """Initialise.

        Args:
            ckpt_base_name: Base name to give all checkpoint files.
            ckpt_location: Location where checkpoint file are stored. Default to the
                local filesystem.
            overwrite_existing: Whether to overwrite existing checkpoints. Defaults to
                False.

        """
        self.ckpt_base_name = ckpt_base_name
        self.ckpt_location = ckpt_location

    def save_checkpoint(
            self,
            model: nn.Module,
            optimiser: nn.Module | None,
            step: int,
            extra_metadata: dict[str, Any]
        ) -> None:
        """Save checkpoint to chosen location.

        Args:
            model: The model with state dict to be persisted.
            optimiser: The optimiser with state dict to be persisted (optional).
            step: Training step that produced the model and optimiser.
            extra_metadata: Dictionary of additional related information to be persisted
                with model and optimiser.

        """
        match self.ckpt_location:
            case "LocalFS":
                pass
            case _:
                raise NotImplementedError

    def load_checkpoint(
            self,
            model: nn.Module,
            optimiser: nn.Module | None,
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
        match self.ckpt_location:
            case "LocalFS":
                state_dict = self._load_ckpt_local_fs("foo-1980")
            case _:
                raise NotImplementedError

        ckpt = Checkpoint(
            state_dict["model"], state_dict.get("optimiser"), state_dict["metadata"]
        )
        return ckpt

    def list_checkpoints(self) -> list[str]:
        """Get list of all checkpoints with base name.

        Returns:
            List of all checkpoint associated with the base name.

        """
        match self.ckpt_location:
            case "LocalFS":
                return self._list_ckpts_local_fs()
            case _:
                raise NotImplementedError

    @staticmethod
    def _save_ckpt_local_fs(state_dict: dict[str, Any], ckpt_name) -> None:
        """Save checkpoint to local FS.

        Args:
            state_dict: State dict to persist to local FS.
            ckpt_name: Full name of checkpoint.

        """
        pass


    @staticmethod
    def _load_ckpt_local_fs(ckpt_name) -> dict[str, Any]:
        """Load checkpoint from local FS.

        Args:
            state_dict: State dict to persist to local FS.
            ckpt_name: Full name of checkpoint.

        Returns:
            State dict for model, optimiser and metadata.

        """
        return {"foo": "bar"}

    def _list_ckpts_local_fs(self) -> list[str]:
        """Get list of checkpoint on local FS.

        Returns:
            List of all checkpoint associated with the base name on the local FS.

        """
        return [""]
