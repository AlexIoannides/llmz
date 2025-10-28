"""Tests for LLM evaluation framework."""

import logging
from collections.abc import Callable

from _pytest.logging import LogCaptureFixture
from torch import nn
from torch.utils.data import DataLoader

from llmz.evaluate import EvalResult, Evaluator, basic_llm_metrics
from llmz.train import log


def test_Evaluator_computes_evaluation_metrics(
    model: nn.Module, dataloader: DataLoader, eval_metrics_fn: Callable
):
    eval = Evaluator(dataloader, dataloader, eval_metrics_fn)
    eval.evaluate(1, model)
    eval.evaluate(2, model)
    assert len(eval._eval_records) == 2
    assert eval._eval_records[0].step == 1
    assert eval._eval_records[0].results == {"train_loss": 0.1, "val_loss": 0.1}


def test_Evaluator_computes_evaluation_scenarios(
    model: nn.Module,
    dataloader: DataLoader,
    eval_metrics_fn: Callable,
    eval_scenarios_fn: Callable,
):
    eval = Evaluator(dataloader, dataloader, eval_metrics_fn, eval_scenarios_fn)
    eval.evaluate(1, model)
    eval.evaluate(2, model)
    assert eval._eval_records[0].results["sample_text"] == "I've seen things..."
    assert eval._eval_records[1].results["sample_text"] == "I've seen things..."


def test_Evaluator_logs_evaluations(
    caplog: LogCaptureFixture, model: nn.Module, dataloader: DataLoader, eval_metrics_fn
):
    eval = Evaluator(dataloader, dataloader, eval_metrics_fn)
    with caplog.at_level(logging.INFO):
        eval.evaluate(1, model, log)
    assert "train_loss=0.1" in caplog.text
    assert "val_loss=0.1" in caplog.text


def test_Evaluator_iter(
    model: nn.Module,
    dataloader: DataLoader,
    eval_metrics_fn: Callable,
):
    eval = Evaluator(dataloader, dataloader, eval_metrics_fn)
    eval.evaluate(1, model)
    eval.evaluate(2, model)

    count = 0
    for n, e in enumerate(eval):
        assert isinstance(e, EvalResult)
        assert isinstance(eval[n], EvalResult)
        count += 1
    assert count == len(eval) == 2


def test_basic_llm_metrics(model: nn.Module, dataloader: DataLoader):
    metrics = basic_llm_metrics(model, dataloader)
    assert isinstance(metrics["loss"], float)
    assert metrics["loss"] > 0.0
