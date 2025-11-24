"""Functions for training LLMs."""

import logging
import math
import sys
from collections.abc import Callable, Generator

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from llmz.checkpoint_handlers import _CheckpointHandler
from llmz.evaluate import Evaluator

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
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


class GradientClipCallback:
    """Callable class that clips model gradient using max norm."""

    def __init__(self, clip_grad_norm: float = torch.inf):
        """Initialise."""
        self.clip_grad_norm = clip_grad_norm

    def __call__(self, model: nn.Module) -> None:
        """Clip model gradients."""
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad_norm)


class TrainLoopManager:
    """TODO."""

    def __init__(
            self,
            epochs: int,
            steps_per_epoch: int,
            start_from_step: int = 1
        ):
        """Initialise."""
        if start_from_step < 1:  # TODO other validations
            raise ValueError("start_from_step must be >= 1")

        self._epoch_step_generator = self._build_epoch_step_generator(
            epochs, steps_per_epoch, start_from_step
        )

    def __iter__(self) -> "TrainLoopManager":
        return self

    def __next__(self) -> tuple[int, int]:
        return next(self._epoch_step_generator)

    @staticmethod
    def _build_epoch_step_generator(
            epochs: int, steps_per_epoch: int, current_step: int
        ) -> Generator[tuple[int, int]]:
        """TODO."""
        total_steps = epochs * steps_per_epoch
        num_remaining_steps_ex_current = total_steps - current_step

        steps_left_in_current_epoch = num_remaining_steps_ex_current % steps_per_epoch
        num_whole_epochs_remaining = num_remaining_steps_ex_current // steps_per_epoch
        current_epoch = (epochs - num_whole_epochs_remaining)

        if steps_left_in_current_epoch > 0:
            for _ in range(steps_per_epoch):
                yield (current_epoch, current_step)
                current_step += 1
            current_epoch += 1

        for epoch in range(current_epoch, epochs):
            for _ in range(steps_per_epoch):
                yield (epoch, current_step)
                current_step += 1


def train(
    model: nn.Module,
    loss_calc: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    optimiser: optim.Optimizer,
    lr_schedule: Callable[[int], float] | lr_scheduler.LRScheduler,
    train_dataloader: DataLoader,
    train_epochs: int,
    evaluator: Evaluator,
    eval_ckpt_freq_steps: int,
    ckpt_handler: _CheckpointHandler | None = None,
    model_backward_callbacks: list[Callable[[nn.Module], None]] | None = None,
    log_freq_steps: int = 100,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Trains model.

    Args:
        model: The PyTorch model to train.
        loss_calc: Function that calculates and returns loss for model and batch.
        optimiser: The optimizer for updating model parameters.
        lr_schedule: Function to compute learning rate for training step.æ
        train_dataloader: DataLoader for training data.
        train_epochs: Number of training epochs.
        evaluator: A handler for all model evaluations.
        eval_ckpt_freq_steps: Number of steps between evaluations and checkpoint
            persistence.
        ckpt_handler: Optional checkpoint handler for persisting checkpoints after
            evaluations have been computed. Defaults to None.
        model_backward_callbacks: Optional callbacks for model after backward pass.
        log_freq_steps: Number of steps between basic progress logging to stdout.
            Defaults to 100.
        device: The processor to use for training. Defaults to CPU.

    """
    if not isinstance(lr_schedule, lr_scheduler.LRScheduler):
        lr_schedule = lr_scheduler.LambdaLR(optimiser, lr_schedule)

    model = model.to(device)
    step = 0

    for epoch in range(1, train_epochs + 1):
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

            if step % eval_ckpt_freq_steps == 0:
                evaluator.evaluate(step, model, log)
                if ckpt_handler:
                    ckpt_handler.save_checkpoint(
                        model,
                        optimiser,
                        lr_schedule,
                        step,
                        {"evals": evaluator[-1].results},
                    )


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

    # flatten logits from [BATCH, SEQ_LEN, N_CLASSES] to [BATCH * SEQ_LEN, N_CLASSES]
    # flatten y_batch from [BATCH, SEQ_LEN] to [BATCH * SEQ_LEN]
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), y_batch.flatten())
    return loss
