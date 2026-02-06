# -*- coding: utf-8 -*-
"""Configuration classes for context pruning."""

from dataclasses import dataclass, field
from typing import Literal, List


@dataclass
class SoftTrimConfig:
    """Configuration for soft-trimming large tool results."""

    max_chars: int = 4000
    """Maximum characters before soft-trimming is applied."""

    head_chars: int = 1500
    """Number of characters to keep from the head."""

    tail_chars: int = 1500
    """Number of characters to keep from the tail."""


@dataclass
class HardClearConfig:
    """Configuration for hard-clearing very large tool results."""

    enabled: bool = True
    """Whether hard-clearing is enabled."""

    placeholder: str = "[Old tool result content cleared]"
    """The placeholder text to use when clearing."""


@dataclass
class ToolPruningConfig:
    """Configuration for tool-specific pruning rules."""

    allow: List[str] = field(default_factory=lambda: ["*"])
    """List of tool name patterns to allow pruning (supports wildcards)."""

    deny: List[str] = field(default_factory=list)
    """List of tool name patterns to deny pruning (supports wildcards).
    Deny rules override allow rules."""


@dataclass
class PruningConfig:
    """Configuration for context pruning.

    Context pruning reduces the size of tool results in the message history
    before sending to the LLM, helping to stay within token limits and reduce
    costs, especially when using providers with prompt caching.
    """

    mode: Literal["off", "cache-ttl", "always"] = "off"
    """Pruning mode:
    - "off": No pruning
    - "cache-ttl": Only prune if last API call is older than ttl
    - "always": Prune on every API call
    """

    ttl: str = "5m"
    """Time-to-live for cache-ttl mode. Format: "5m", "1h", "30s"."""

    keep_last_assistants: int = 3
    """Number of most recent assistant messages to protect from pruning."""

    soft_trim_ratio: float = 0.3
    """Context usage ratio to trigger soft-trimming (0.0-1.0)."""

    hard_clear_ratio: float = 0.5
    """Context usage ratio to trigger hard-clearing (0.0-1.0)."""

    min_prunable_tool_chars: int = 50_000
    """Minimum total tool result characters before pruning activates."""

    soft_trim: SoftTrimConfig = field(default_factory=SoftTrimConfig)
    """Configuration for soft-trimming."""

    hard_clear: HardClearConfig = field(default_factory=HardClearConfig)
    """Configuration for hard-clearing."""

    tools: ToolPruningConfig = field(default_factory=ToolPruningConfig)
    """Tool-specific pruning rules."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.soft_trim_ratio < 0 or self.soft_trim_ratio > 1:
            raise ValueError(
                "soft_trim_ratio must be between 0.0 and 1.0",
            )
        if self.hard_clear_ratio < 0 or self.hard_clear_ratio > 1:
            raise ValueError(
                "hard_clear_ratio must be between 0.0 and 1.0",
            )
        if self.keep_last_assistants < 0:
            raise ValueError(
                "keep_last_assistants must be non-negative",
            )
