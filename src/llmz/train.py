"""Functions for training LLMs."""

import logging
import math
import sys
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("llmz.train")


class LinearWarmupCosineAnnealingLRSchedule:
    """LR schedule using cosine annealing with linear warmup."""

    def __init__(
            self, num_steps: int, warmup_steps: int, initial_lr: float, peak_lr: float
        ):
        """Initialise.

        Args:
            num_steps: The total number of steps for the schedule.
            warmup_steps: Number of steps in the linear warmup phase.
            initial_lr: Learning rate at first step.
            peak_lr: Peak learning rate at end of warmup phase.

        """
        value_errors: list[str] = []
        if num_steps <= 0:
            value_errors.append(" * num_steps <= 0")
        if warmup_steps > num_steps:
            value_errors.append(" * warmup_steps > num_steps")
        if initial_lr <= 0.0:
            value_errors.append(" * initial_lr <= 0.0")
        if peak_lr < initial_lr:
            value_errors.append(" * peak_lr < initial_lr")

        if value_errors:
            e = ValueError("Invalid arguments for LR schedule")
            for error in value_errors:
                e.add_note(error)
            raise e

        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        self.cosine_steps = num_steps - warmup_steps
        self.initial_lr = initial_lr
        self.lr_cosine_delta = peak_lr - initial_lr
        self.lr_warmup_delta = (peak_lr - initial_lr) / warmup_steps

    def __call__(self, step: int) -> float:
        """Get learning rate for given step.

        Args:
            step: The global training step.

        Returns:
            The learning rate for the global training step.

        Raises:
            ValueError: If step < 0.

        """
        if step < 0:
            raise ValueError(f"{step=}, must be > 0")
        elif step >= 0 and step < self.warmup_steps:
            lr = self.initial_lr + step * self.lr_warmup_delta
        elif step >= self.warmup_steps and step <= self.num_steps:
            step_cosine = step - self.warmup_steps
            x = math.pi * step_cosine / self.cosine_steps
            lr = self.initial_lr + self.lr_cosine_delta * 0.5 * (1.0 + math.cos(x))
        else:
            lr = self.initial_lr

        return lr


class EvalResult(NamedTuple):
    """Container for evaluation results produced during training."""

    step: int
    results: dict[str, float | int | str]


class Evaluator:
    """Model evaluator.

    This class executes and stores all model evaluations during training.
    """

    def __init__(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Initialise.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.

        """
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self._eval_records: list[EvalResult]

    def evaluate(
            self, step: int, model: nn.Module, log: logging.Logger | None = log
        ) -> None:
        """Evaluate model.

        Args:
            step: The number of training steps applied to the model.
            model: The model to evaluate.
            log: Optional logger for logging results? Defaults to custom llmz logger.

        Return:
            All evaluations for the model after training steps.

        """
        pass

    @staticmethod
    def _compute_metrics(model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
        """Compute all metrics for a dataloader."""
        return {"A": 1.0}

    @staticmethod
    def _compute_scenarios(model: nn.Module) -> dict[str, float | int | str]:
        """Compute model output for specific input scenarios."""
        return {"A": 1.0}


class GradientClipCallback:
    """Callable class that clips model gradient using max norm."""

    def __init__(self, clip_grad_norm: float = torch.inf):
        """Initialise."""
        self.clip_grad_norm = clip_grad_norm

    def __call__(self, model: nn.Module) -> None:
        """Clip model gradients."""
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad_norm)


def train(
    model: nn.Module,
    loss_calc: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    optimiser: optim.Optimizer,
    lr_schedule: Callable[[int], float] | optim.lr_scheduler.LRScheduler,
    train_dataloader: DataLoader,
    train_epochs: int,
    eval_freq_steps: int,
    evaluator: Evaluator,
    model_backward_callbacks: list[Callable[[nn.Module], None]] | None = None,
    log_freq_steps: int = 100,
    device: torch.device = torch.device("cpu"),
    ) -> None:
    """Trains model.

    Args:
        model: The PyTorch model to train.
        loss_calc: Function that calculates and returns loss for model and batch.
        optimiser: The optimizer for updating model parameters.
        lr_schedule: Function to compute learning rate for training step.
        train_dataloader: DataLoader for training data.
        train_epochs: Number of training epochs.
        eval_freq_steps: Number of steps between evaluations.
        evaluator: A handler for all model evaluations.
        model_backward_callbacks: Optional callbacks for model after backward pass.
        log_freq_steps: Number of steps between basic progress logging to stdout.
            Defaults to 100.
        device: The processor to use for training. Defaults to CPU.

    """
    if not isinstance(lr_schedule, optim.lr_scheduler.LRScheduler):
        lr_schedule = optim.lr_scheduler.LambdaLR(optimiser, lr_schedule)

    model = model.to(device)
    step = 0

    for epoch in range(1, train_epochs+1):
        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            step += 1
            model.train()
            optimiser.zero_grad()

            loss = loss_calc(model, X_batch, y_batch)
            loss.backward()

            if model_backward_callbacks:
                for callback in model_backward_callbacks:
                    callback(model)

            optimiser.step()
            lr_schedule.step()

            if step % log_freq_steps == 0:
                log.info(f"{step=}, {epoch=}")

            if step % eval_freq_steps == 0:
                evaluator.evaluate(step, model)


def autoregressive_llm_loss(
        model: nn.Module, X_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> torch.Tensor:
    """Compute loss for AR LLMs like GPTs.

    Args:
        model: The language model.
        X_batch: Batch of input tokens.
        y_batch: Batch of output tokens - i.e., next token from the input sequence.

    Returns:
        Mean cross-entropy loss for the batch.

    """
    # model outputs logits as softmax is implemented in cross-entropy calc.
    logits = model(X_batch)

    # flatten logits from [BATCH, SEQ_LEN, N_CLASSES] to [BATCH*SEQ_LEN, N_CLASSES]
    # flatten y_batch from [BATCH, SEQ_LEN] to [BATCH * SEQ_LEN]
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), y_batch.flatten())
    return loss
