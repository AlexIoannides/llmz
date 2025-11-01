"""Functional for end-to-end training."""

from pathlib import Path

from torch.optim import AdamW

from llmz.datasets import GPTSmallTextDataset
from llmz.evaluate import Evaluator, basic_llm_metrics
from llmz.gpt2 import GPT2, GPT2Config
from llmz.train import (
    LinearWarmupCosineAnnealingLRSchedule,
    autoregressive_llm_loss,
    train,
)


def test_GPT2_train_end_to_end(text_data_file: Path):
    context_size = 256
    warmup_epochs = 2
    total_epochs = 12
    lr = 0.001
    batch_size=8

    train_ds = GPTSmallTextDataset(text_data_file.read_text(), context_size)
    train_dl = train_ds.create_data_loader(batch_size)

    model_config = GPT2Config(
        vocab_size=train_ds.vocab_size,
        embed_dim=256,
        context_size=context_size,
        n_tsfmr_blocks=1, n_attn_heads=4,
    )

    model = GPT2(**model_config)

    evals = Evaluator(
        train_dataloader=train_dl, val_dataloader=train_dl, metrics_fn=basic_llm_metrics
    )

    steps_per_epoch = int(len(train_ds) / batch_size)
    lr_schedule = LinearWarmupCosineAnnealingLRSchedule(
        num_steps=total_epochs*steps_per_epoch,
        warmup_steps=warmup_epochs*steps_per_epoch,
        initial_lr=lr/100,
        peak_lr=lr,
    )

    optim = AdamW(model.parameters())

    train(
        model=model,
        loss_calc=autoregressive_llm_loss,
        optimiser=optim,
        lr_schedule=lr_schedule,
        train_dataloader=train_dl,
        train_epochs=total_epochs,
        evaluator=evals,
        eval_freq_steps=steps_per_epoch,
        log_freq_steps=steps_per_epoch,
    )

