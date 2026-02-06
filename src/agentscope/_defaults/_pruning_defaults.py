# -*- coding: utf-8 -*-
"""Smart defaults for context pruning configuration."""

from ..memory._pruning import PruningConfig


def get_anthropic_pruning_defaults(
    cache_aware: bool = True,
) -> PruningConfig:
    """Get smart defaults for Anthropic models.

    Anthropic models support prompt caching, so we enable cache-ttl mode
    to avoid re-caching the full context when the cache expires.

    Args:
        cache_aware (`bool`, defaults to `True`):
            Whether to use cache-aware pruning (cache-ttl mode).
            If False, uses "always" mode.

    Returns:
        `PruningConfig`:
            The pruning configuration optimized for Anthropic models.
    """
    return PruningConfig(
        mode="cache-ttl" if cache_aware else "always",
        ttl="5m",  # Match typical Anthropic cache TTL
        keep_last_assistants=3,
        soft_trim_ratio=0.3,
        hard_clear_ratio=0.5,
        min_prunable_tool_chars=50_000,
    )


def get_openai_pruning_defaults() -> PruningConfig:
    """Get smart defaults for OpenAI models.

    OpenAI models don't have prompt caching (yet), so we use a more
    conservative approach with less aggressive pruning.

    Returns:
        `PruningConfig`:
            The pruning configuration optimized for OpenAI models.
    """
    return PruningConfig(
        mode="always",
        keep_last_assistants=5,  # Keep more recent messages
        soft_trim_ratio=0.5,  # Less aggressive
        hard_clear_ratio=0.7,
        min_prunable_tool_chars=75_000,
    )


def get_gemini_pruning_defaults() -> PruningConfig:
    """Get smart defaults for Google Gemini models.

    Gemini has very large context windows, so we prune less aggressively.

    Returns:
        `PruningConfig`:
            The pruning configuration optimized for Gemini models.
    """
    return PruningConfig(
        mode="always",
        keep_last_assistants=10,  # Larger context window
        soft_trim_ratio=0.6,
        hard_clear_ratio=0.8,
        min_prunable_tool_chars=100_000,
    )


def get_disabled_pruning() -> PruningConfig:
    """Get a pruning configuration with pruning disabled.

    Returns:
        `PruningConfig`:
            A configuration with pruning turned off.
    """
    return PruningConfig(mode="off")
