"""Tools for text generation ."""

from typing import Literal

import torch


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
