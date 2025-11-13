"""Functional for end-to-end training."""

import shutil
from collections.abc import Generator
from pathlib import Path
from typing import cast

import pytest
import torch
from torch.optim import AdamW

from llmz.checkpoint_handlers import LOCAL_FS_PATH, LocalFSCheckpointHandler
from llmz.datasets import GPT2SmallTextDataset
from llmz.evaluate import Evaluator, basic_llm_metrics
from llmz.generate import generate
from llmz.gpt2 import GPT2, GPT2Config, GPT2Tokenizer
from llmz.train import (
    LinearWarmupCosineAnnealingLRSchedule,
    autoregressive_llm_loss,
    train,
)

CKPT_SUB_DIR = "functional_test"


@pytest.fixture
def cleanup_ckpt_dir() -> Generator[None, None]:
    shutil.rmtree(LOCAL_FS_PATH / CKPT_SUB_DIR, ignore_errors=True)
    yield None
    shutil.rmtree(LOCAL_FS_PATH / CKPT_SUB_DIR, ignore_errors=True)


@pytest.mark.usefixtures("cleanup_ckpt_dir")
def test_GPT2_train_end_to_end(text_data_file: Path):
    torch.manual_seed(1980)

    context_size = 256
    warmup_epochs = 2
    total_epochs = 10
    lr = 0.005
    batch_size = 8
    device = torch.device("cpu")

    train_ds = GPT2SmallTextDataset(text_data_file.read_text(), context_size)
    train_dl = train_ds.create_data_loader(batch_size, num_workers=1)

    steps_per_epoch = int(len(train_ds) / batch_size)

    model_config = GPT2Config(
        vocab_size=train_ds.tokenizer.vocab_size,
        embed_dim=256,
        context_size=context_size,
        n_tsfmr_blocks=1,
        n_attn_heads=4,
    )

    model = GPT2(**model_config)

    evals = Evaluator(
        train_dataloader=train_dl,
        val_dataloader=train_dl,
        metrics_fn=basic_llm_metrics,
        device=device,
    )

    checkpointer = LocalFSCheckpointHandler(CKPT_SUB_DIR)

    lr_schedule = LinearWarmupCosineAnnealingLRSchedule(
        num_steps=total_epochs * steps_per_epoch,
        warmup_steps=warmup_epochs * steps_per_epoch,
        initial_lr=lr / 100,
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
        eval_ckpt_freq_steps=steps_per_epoch,
        ckpt_handler=checkpointer,
        log_freq_steps=steps_per_epoch,
        device=device,
    )

    train_loss_beginning = cast(float, evals[0].results["train_loss"])
    train_loss_end = cast(float, evals[-1].results["train_loss"])
    assert train_loss_beginning > train_loss_end

    prompt = "I've seen things you people wouldn't believe"
    generated_text = generate(
        model=model,
        prompt=prompt,
        tokenizer=GPT2Tokenizer(),
        strategy="greedy",
        output_length=60,
        device=device,
    )
    assert len(generated_text) > len(prompt)

    ckpt_dir = Path(".llmz_ckpts") / CKPT_SUB_DIR
    ckpts = list(ckpt_dir.glob("*.pt"))
    assert len(ckpts) == 10

    last_ckpt = checkpointer.load_checkpoint(
        model,
        optim,
        torch.optim.lr_scheduler.LambdaLR(optim, lr_schedule),
    )
    assert last_ckpt.step == total_epochs * steps_per_epoch
    assert last_ckpt.metadata is not None
    assert last_ckpt.metadata["evals"]["train_loss"] == train_loss_end
