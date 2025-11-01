"""Evaluation and metrics."""

import logging
from collections.abc import Callable, Iterator
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import DataLoader

MetricResult = float | int
MetricsFunc = Callable[[nn.Module, DataLoader, torch.device], dict[str, MetricResult]]

ScenarioResult = float | int | str
ScenariosFunc = Callable[[nn.Module, torch.device], dict[str, ScenarioResult]]


class EvalResult(NamedTuple):
    """Container for evaluation results produced during training."""

    step: int
    results: dict[str, MetricResult | ScenarioResult]


class Evaluator:
    """Model evaluator.

    This class executes and stores all model evaluations during training.
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        metrics_fn: MetricsFunc,
        scenarios_fn: ScenariosFunc | None = None,
        device: torch.device = torch.device("cpu"),
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
            device: The processor to use for training. Defaults to CPU.

        """
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.metrics_fn = metrics_fn
        self.scenarios_fn = scenarios_fn
        self._eval_records: list[EvalResult] = []
        self.device = device

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
            f"train_{k}": v
            for k, v in self.metrics_fn(model, self.train_dl, self.device).items()
        }
        val_metrics = {
            f"val_{k}": v
            for k, v in self.metrics_fn(model, self.val_dl, self.device).items()
        }
        scenarios = self.scenarios_fn(model, self.device) if self.scenarios_fn else {}

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


def basic_llm_metrics(
    model: nn.Module, dl: DataLoader, device=torch.device("cpu")
) -> dict[str, float]:
    """Compute basic LLM metrics for a dataloader.

    Args:
        model: Model to use for inference.
        dl: Dataloader with data batches for inference.
        device: The processor to use for training. Defaults to CPU.

    """
    loss = sum(
        f.cross_entropy(
            model(X.to(device)).flatten(0, 1), y.to(device).flatten()
        ).item()
        for X, y in dl
    ) / len(dl)
    return {"loss": loss}
