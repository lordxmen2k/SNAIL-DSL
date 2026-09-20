"""Pricing tables for hosted LLM providers.

Per-model USD pricing for input and output tokens, last verified
September 2026. Users can override via HostedNode's `cost_per_1k_input`
and `cost_per_1k_output` kwargs.

License: Apache 2.0. Copyright 2026 Tico Internet LLC.
"""

from __future__ import annotations
from typing import Optional


# Last verified: 2026-09-20. Source: vendor pricing pages.
# Update by editing the relevant vendor's section. Prices are USD per
# 1,000 tokens (input / output).

ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (cost_per_1k_input_usd, cost_per_1k_output_usd)
    "claude-haiku-4-5":  (0.001,  0.005),
    "claude-haiku-5":    (0.001,  0.005),
    "claude-sonnet-4-5": (0.003,  0.015),
    "claude-sonnet-5":   (0.003,  0.015),
    "claude-opus-4":     (0.015,  0.075),
    "claude-opus-4-5":   (0.015,  0.075),
    "claude-opus-5":     (0.025,  0.125),
}


OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5":      (0.005,  0.015),
    "gpt-5-mini": (0.0005, 0.0015),
    "gpt-5-nano": (0.0001, 0.0004),
    "gpt-4o":     (0.005,  0.015),
    "gpt-4o-mini":(0.00015,0.0006),
}


def cost_usd(
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    *,
    override_input: Optional[float] = None,
    override_output: Optional[float] = None,
) -> float:
    """Compute the cost in USD for a given provider/model/token counts.

    Args:
        provider: "anthropic", "openai", "ollama", or "stub".
        model: Model identifier (e.g. "claude-sonnet-4-5").
        tokens_in: Input token count.
        tokens_out: Output token count.
        override_input: User-supplied cost per 1k input tokens.
        override_output: User-supplied cost per 1k output tokens.

    Returns:
        Cost in USD. Local providers (ollama, stub) return 0.
    """
    if provider in ("ollama", "stub"):
        return 0.0

    if override_input is not None and override_output is not None:
        in_rate, out_rate = override_input, override_output
    else:
        table = (
            ANTHROPIC_PRICING if provider == "anthropic" else OPENAI_PRICING
        )
        rates = table.get(model)
        if rates is None:
            return 0.0  # Unknown model: cost = 0 (don't make up numbers)
        in_rate, out_rate = rates

    return (tokens_in / 1000.0) * in_rate + (tokens_out / 1000.0) * out_rate
