"""Test for GP2 model."""

import re

import pytest
import torch

from llmz.gpt2 import GPT2, GPT2Config, GPT2ConfigError, GPT2InferenceError, GPT2Tokenizer


@pytest.mark.parametrize(
    "vocab_size, batch_size, context_size, embed_dim, n_tsfmr_blocks",
    [(10, 1, 2, 4, 1), (20, 1, 3, 8, 2), (27, 2, 4, 12, 3)],
)
def test_TransformerBlock_output_properties(
    vocab_size: int,
    batch_size: int,
    context_size: int,
    embed_dim: int,
    n_tsfmr_blocks: int,
):
    tokens_batch = torch.randint(
        0, vocab_size, (batch_size, context_size), dtype=torch.int32
    )

    model = GPT2(vocab_size, embed_dim, context_size, n_tsfmr_blocks)
    out_batch = model(tokens_batch)

    assert out_batch.size() == torch.Size((batch_size, context_size, vocab_size))
    assert torch.all(torch.isreal(out_batch))


def test_TransformerBlock_raises_error_for_incomputable_inference():
    model = GPT2(vocab_size=5, embed_dim=5, context_size=5, n_tsfmr_blocks=1)
    expected_err_msg = re.escape("seq_len (6) > context_size (5)")
    with pytest.raises(GPT2InferenceError, match=expected_err_msg):
        x = torch.ones((1, 6))
        model(x)


def test_GPT2Config_validates_fields():
    # valid config
    try:
        GPT2Config(
            vocab_size=1000,
            embed_dim=800,
            context_size=100,
            n_tsfmr_blocks=4,
            n_attn_heads=8,
            dropout=0.2,
            qkv_bias=False,
        )
        assert True
    except Exception:
        assert False

    # invalid config
    expected_msg = re.escape(
        "invalid GPT2 parameters: vocab_size is not > 0\n embed_dim % n_attn_heads != 0"
    )
    with pytest.raises(GPT2ConfigError, match=expected_msg):
        GPT2Config(
            vocab_size=-1000,
            embed_dim=801,
            context_size=100,
            n_tsfmr_blocks=4,
            n_attn_heads=8,
            dropout=0.2,
            qkv_bias=False,
        )


def test_GPT2Config_kwarg_expansion():
    try:
        config = GPT2Config(
            vocab_size=1000,
            embed_dim=800,
            context_size=100,
            n_tsfmr_blocks=4,
            n_attn_heads=8,
        )
        GPT2(**config)
        assert True
    except Exception:
        assert False


def test_GPT2Config_string_representation():
    config = GPT2Config(
        vocab_size=1000,
        embed_dim=800,
        context_size=100,
        n_tsfmr_blocks=4,
        n_attn_heads=8,
    )
    expected_output_pattern = (
        r"GPT2Config\(([a-zA-Z0-9_]+=([\.\d]+|True|False|\".*\")(,\s|))+\)$"
    )
    assert re.match(expected_output_pattern, str(config)) is not None


def test_GPT2Config_command_line_representation():
    config = GPT2Config(
        vocab_size=1000,
        embed_dim=800,
        context_size=100,
        n_tsfmr_blocks=4,
        n_attn_heads=8,
    )
    expected_output_pattern = (
        r"^GPT2Config\(\n(\s\s[a-zA-Z0-9_]+=([\.\d]+|True|False|\".*\")+,\n)+\)$"
    )
    assert re.match(expected_output_pattern, config.__repr__()) is not None


@pytest.mark.parametrize("text", ["foo bar", "My name is Alex"])
def test_GP2Tokenizer_tokenizes_text(text: str):
    tokenizer = GPT2Tokenizer()
    tokens = tokenizer.text2tokens(text)
    assert len(tokens) > 1
    assert isinstance(tokens[0], int)
    assert tokenizer.tokens2text(tokens) == text
