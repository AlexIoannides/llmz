"""Tools for text generation ."""

import re
from textwrap import wrap
from typing import Literal

import torch
import torch.nn as nn

from llmz.tokenizers import _Tokenizer


def generate(
    model: nn.Module,
    prompt: str,
    tokenizer: _Tokenizer,
    strategy: Literal["greedy", "sample", "topk"] = "greedy",
    output_length: int = 60,
    temperature: float = 1.0,
    random_seed: int = 42,
    device: torch.device = torch.device("cpu"),
    *,
    k: int = 2,
) -> str:
    """Generate new text conditional on a text prompt."""
    torch.manual_seed(random_seed)

    model.to(device)
    model.eval()

    prompt_tokens = tokenizer(prompt)
    token_sequence = prompt_tokens.copy()
    for _ in range(output_length):
        x = torch.tensor([token_sequence], device=device)
        token_logits = model(x)
        token_pred = decode(token_logits[0, -1], strategy, temperature, k=k)
        token_sequence += [token_pred.item()]

    new_token_sequence = token_sequence[len(prompt_tokens) :]
    new_token_sequence = token_sequence[len(prompt_tokens) :]
    return format_generated_words(tokenizer.tokens2text(new_token_sequence), prompt)


def decode(
    token_logits: torch.Tensor,
    strategy: Literal["greedy", "sample", "topk"] = "greedy",
    temperature: float = 1.0,
    *,
    k: int = 5,
) -> torch.Tensor:
    """Decode generative model output using the specified strategy."""
    match strategy:
        case "greedy":
            return _greedy_decoding(token_logits, temperature)
        case "topk":
            return _top_k_decoding(token_logits, temperature, k)
        case "sample":
            return _sample_decoding(token_logits, temperature)


def _sample_decoding(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Generate next token using sample decoding strategy."""
    return torch.distributions.Categorical(
        logits=logits.squeeze() / temperature
    ).sample()


def _top_k_decoding(
    logits: torch.Tensor, temperature: float = 1.0, k: int = 3
) -> torch.Tensor:
    """Generate next token using top-k decoding strategy."""
    token_probs = torch.distributions.Categorical(
        logits=logits.squeeze() / temperature
    ).probs
    top_k_tokens = torch.topk(token_probs, k=k)
    sampled_token = torch.distributions.Categorical(probs=top_k_tokens.values).sample()
    return top_k_tokens.indices[sampled_token]


def _greedy_decoding(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Generate next token using greedy decoding strategy."""
    token_probs = torch.distributions.Categorical(
        logits=logits.squeeze() / temperature
    ).probs
    return torch.argmax(token_probs)


def format_generated_words(text: str, prompt: str) -> str:
    """Format list of words into a readable paragraph."""
    text = _capitalise_sentences(text, sentence_delimiter=". ")
    text = "==> " + prompt.upper().strip() + " " + text.strip()
    return "\n".join([line for line in wrap(text, width=89)])


def _capitalise_sentences(text: str, sentence_delimiter: str = ". ") -> str:
    """Capitalise the first letter of sentences in text passage."""
    sentences = text.split(sentence_delimiter)
    sentences = [sentence[:1].upper() + sentence[1:] for sentence in sentences]
    return sentence_delimiter.join(sentences)


def print_wrapped(text: str, width: int = 89) -> None:
    """Print text with word wrapping."""
    wrapped_text = "\n".join(wrap(text, width=width))
    print(wrapped_text)
