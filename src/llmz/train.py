"""Functions for training LLMs."""

import math
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


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
            value_errors.append(" num_steps <= 0")
        if warmup_steps > num_steps:
            value_errors.append(" warmup_steps > num_steps")
        if initial_lr <= 0.0:
            value_errors.append(" initial_lr <= 0.0")
        if peak_lr < initial_lr:
            value_errors.append(" peak_lr < initial_lr")

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


class Evaluator:
    """Model evaluator.

    This class executes all model evaluations and stores the results.
    """

    def __init__(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Initialise.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.

        """
        self._eval_records: list[dict[str, float | int | str | bool]] = []

    def __call__(self, step: int, model: nn.Module) -> None:
        """Evaluate model.

        Args:
            step: The number of training steps applied to the model.
            model: The model to evaluate.

        """
        pass

    @staticmethod
    def _compute_metrics(model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
        """Compute all metrics for a dataloader."""
        return {"A": 1.0}

    @staticmethod
    def _compute_scenarios(model: nn.Module) -> dict[str, float | int | str | bool]:
        """Compute predictions for specific scenarios."""
        return {"A": 1.0}


def train(
        model: nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        train_epochs: int,
        eval_freq_steps: int,
        lr_schedule: Callable[[int], float],
        evaluator: Evaluator,
    ) -> None:
    """Trains model.

    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer for updating model parameters.
        train_dataloader: DataLoader for training data.
        train_epochs: Number of training epochs.
        val_dataloader: DataLoader for validation data.
        val_freq_steps: Number of steps between validations.
        lr_schedule: Function to compute learning rate for training step.

    """
    pass


# from previous_chapters import evaluate_model, generate_and_print_sample
# # Alternatively:
# # from llms_from_scratch.ch05 import evaluate_model, generate_and_print_samplee


# ORIG_BOOK_VERSION = False


# def train_model(model, train_loader, val_loader, optimizer, device,
#                 n_epochs, eval_freq, eval_iter, start_context, tokenizer,
#                 warmup_steps, initial_lr=3e-05, min_lr=1e-6):

#     train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
#     tokens_seen, global_step = 0, -1

#     # Retrieve the maximum learning rate from the optimizer
#     peak_lr = optimizer.param_groups[0]["lr"]

#     # Calculate the total number of iterations in the training process
#     total_training_steps = len(train_loader) * n_epochs

#     # Calculate the learning rate increment during the warmup phase
#     lr_increment = (peak_lr - initial_lr) / warmup_steps

#     for epoch in range(n_epochs):
#         model.train()
#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()
#             global_step += 1

#             # Adjust the learning rate based on the current phase (warmup or cosine annealing)
#             if global_step < warmup_steps:
#                 # Linear warmup
#                 lr = initial_lr + global_step * lr_increment  
#             else:
#                 # Cosine annealing after warmup
#                 progress = ((global_step - warmup_steps) / 
#                             (total_training_steps - warmup_steps))
#                 lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

#             # Apply the calculated learning rate to the optimizer
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = lr
#             track_lrs.append(lr)  # Store the current learning rate

#             # Calculate and backpropagate the loss
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()

#             # Apply gradient clipping after the warmup phase to avoid exploding gradients
#             if ORIG_BOOK_VERSION:
#                 if global_step > warmup_steps:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
#             else:
#                 if global_step >= warmup_steps:  # the book originally used global_step > warmup_steps, which led to a skipped clipping step after warmup
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
#             optimizer.step()
#             tokens_seen += input_batch.numel()

#             # Periodically evaluate the model on the training and validation sets
#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader,
#                     device, eval_iter
#                 )
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 # Print the current losses
#                 print(f"Ep {epoch+1} (Iter {global_step:06d}): "
#                       f"Train loss {train_loss:.3f}, "
#                       f"Val loss {val_loss:.3f}"
#                 )

#         # Generate and print a sample from the model to monitor progress
#         generate_and_print_sample(
#             model, tokenizer, device, start_context
#         )

#     return train_losses, val_losses, track_tokens_seen, track_lrs
