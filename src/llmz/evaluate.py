"""Evaluation and metrics."""

import logging
from collections.abc import Callable, Iterator
from typing import NamedTuple

from torch import nn
from torch.nn import functional as f
from torch.utils.data import DataLoader

Result = float | int | str


class EvalResult(NamedTuple):
    """Container for evaluation results produced during training."""

    step: int
    results: dict[str, Result]


class Evaluator:
    """Model evaluator.

    This class executes and stores all model evaluations during training.
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metrics_fn: Callable[[nn.Module, DataLoader], dict[str, Result]],
        scenarios_fn: Callable[[nn.Module], dict[str, Result]] | None = None,
    ):
        """Initialise.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            metrics_fn: Callable that returns a dictionary of metrics given a model and
                a dataloader.
            scenarios_fn: Optional callable that returns a dictionary of results/outputs
                given a model - e.g., generated text given an example prompt. Defaults
                to None.

        """
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.metrics_fn = metrics_fn
        self.scenarios_fn = scenarios_fn
        self._eval_records: list[EvalResult] = []

    def evaluate(
        self, step: int, model: nn.Module, log: logging.Logger | None = None
    ) -> None:
        """Evaluate model.

        Args:
            step: The number of training steps applied to the model.
            model: The model to evaluate.
            log: Optional logger for logging results? Defaults to custom llmz logger.

        Return:
            All evaluations for the model after training steps.

        """
        train_metrics = {
            f"train_{k}": v for k, v in self.metrics_fn(model, self.train_dl).items()
        }
        val_metrics = {
            f"val_{k}": v for k, v in self.metrics_fn(model, self.val_dl).items()
        }
        scenarios = self.scenarios_fn(model) if self.scenarios_fn else {}

        eval_record = EvalResult(step, {**train_metrics, **val_metrics, **scenarios})
        self._eval_records.append(eval_record)

        if log:
            log_msg = f"{eval_record.step=}: " + ", ".join(
                f"{k}={v}" for k, v in eval_record.results.items()
            )
            log.info(log_msg)

    def __getitem__(self, idx: int) -> EvalResult:
        return self._eval_records[idx]

    def __iter__(self) -> Iterator[EvalResult]:
        return iter(self._eval_records)

    def __len__(self) -> int:
        return len(self._eval_records)


def basic_llm_metrics(model: nn.Module, dl: DataLoader) -> dict[str, float]:
    """Compute basic LLM metrics for a dataloader.

    Args:
        model: Model to use for inference.
        dl: Dataloader with data batches for inference.

    """
    loss = sum(
        f.cross_entropy(model(X).flatten(0, 1), y.flatten()).item() for X, y in dl
    ) / len(dl)
    return {"loss": loss}
