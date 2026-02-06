# -*- coding: utf-8 -*-
"""Main context pruner implementation."""

import time
from typing import List, Optional, Set

from .._logging import logger
from ...message import Msg
from ._config import PruningConfig
from ._estimator import (
    estimate_context_window_chars,
    estimate_message_chars,
    estimate_messages_total_chars,
    parse_duration_to_seconds,
)
from ._strategies import (
    soft_trim_tool_result,
    hard_clear_tool_result,
    is_tool_prunable,
    extract_tool_name_from_message,
)


class ContextPruner:
    """Prunes context messages to fit within token limits.

    Implements OpenClaw-style multi-strategy pruning:
    - Soft-trim: Keep head + tail of large tool results
    - Hard-clear: Replace very large tool results with placeholder
    """

    def __init__(self, config: PruningConfig) -> None:
        """Initialize the context pruner.

        Args:
            config (`PruningConfig`):
                The pruning configuration.
        """
        self.config = config

    def should_prune(
        self,
        last_api_call_time: Optional[float] = None,
    ) -> bool:
        """Check if pruning should be activated.

        Args:
            last_api_call_time (`Optional[float]`):
                Timestamp of last API call (for cache-ttl mode).

        Returns:
            `bool`:
                True if pruning should run.
        """
        if self.config.mode == "off":
            return False

        if self.config.mode == "always":
            return True

        if self.config.mode == "cache-ttl":
            if last_api_call_time is None:
                # First call, no need to prune
                return False

            ttl_seconds = parse_duration_to_seconds(self.config.ttl)
            elapsed = time.time() - last_api_call_time

            return elapsed >= ttl_seconds

        return False

    async def prune_messages(
        self,
        messages: List[Msg],
        model_context_window: int = 200_000,
        last_api_call_time: Optional[float] = None,
    ) -> List[Msg]:
        """Prune messages to fit within context window.

        Args:
            messages (`List[Msg]`):
                The messages to prune.
            model_context_window (`int`):
                The model's context window in tokens.
            last_api_call_time (`Optional[float]`):
                Timestamp of last API call.

        Returns:
            `List[Msg]`:
                The pruned messages (original list if no pruning needed).
        """
        if not self.should_prune(last_api_call_time):
            return messages

        if not messages:
            return messages

        # Calculate context window in characters
        char_window = estimate_context_window_chars(model_context_window)

        # Find cutoff index (protect last N assistant messages)
        cutoff_index = self._find_assistant_cutoff_index(
            messages,
            self.config.keep_last_assistants,
        )

        if cutoff_index is None:
            # Not enough assistant messages to establish cutoff
            return messages

        # Identify prunable tool result messages
        prunable_indexes = self._find_prunable_tool_indexes(
            messages,
            cutoff_index,
        )

        if not prunable_indexes:
            return messages

        # Calculate total prunable characters
        prunable_chars = sum(
            estimate_message_chars(messages[i]) for i in prunable_indexes
        )

        if prunable_chars < self.config.min_prunable_tool_chars:
            logger.debug(
                "Prunable tool chars (%d) below threshold (%d), "
                "skipping pruning",
                prunable_chars,
                self.config.min_prunable_tool_chars,
            )
            return messages

        # Calculate current context usage
        total_chars = estimate_messages_total_chars(messages)
        usage_ratio = total_chars / char_window if char_window > 0 else 0

        logger.debug(
            "Context usage: %d chars / %d chars (%.1f%%)",
            total_chars,
            char_window,
            usage_ratio * 100,
        )

        # Apply soft-trimming first
        messages_after_soft_trim = self._apply_soft_trimming(
            messages,
            prunable_indexes,
            usage_ratio,
        )

        # Recalculate usage after soft-trimming
        total_chars_after_soft = estimate_messages_total_chars(
            messages_after_soft_trim,
        )
        usage_ratio_after_soft = (
            total_chars_after_soft / char_window
            if char_window > 0
            else 0
        )

        # Apply hard-clearing if still needed
        if usage_ratio_after_soft >= self.config.hard_clear_ratio:
            return self._apply_hard_clearing(
                messages_after_soft_trim,
                prunable_indexes,
                usage_ratio_after_soft,
            )

        return messages_after_soft_trim

    def _find_assistant_cutoff_index(
        self,
        messages: List[Msg],
        keep_last: int,
    ) -> Optional[int]:
        """Find the cutoff index before which messages can be pruned.

        Args:
            messages (`List[Msg]`):
                The messages.
            keep_last (`int`):
                Number of assistant messages to protect.

        Returns:
            `Optional[int]`:
                The cutoff index, or None if insufficient messages.
        """
        if keep_last <= 0:
            return len(messages)

        assistant_count = 0
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant":
                assistant_count += 1
                if assistant_count >= keep_last:
                    return i

        # Not enough assistant messages
        return None

    def _find_prunable_tool_indexes(
        self,
        messages: List[Msg],
        cutoff_index: int,
    ) -> List[int]:
        """Find indexes of messages that can be pruned.

        Args:
            messages (`List[Msg]`):
                The messages.
            cutoff_index (`int`):
                The cutoff index.

        Returns:
            `List[int]`:
                List of prunable message indexes.
        """
        prunable = []

        for i in range(cutoff_index):
            msg = messages[i]

            # Only prune tool result messages
            if msg.role != "user":
                continue

            # Check if it's a tool result message
            has_tool_result = any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in msg.content
            )

            if not has_tool_result:
                continue

            # Check tool-specific rules
            tool_name = extract_tool_name_from_message(msg)
            if tool_name and not is_tool_prunable(
                tool_name,
                self.config.tools,
            ):
                continue

            prunable.append(i)

        return prunable

    def _apply_soft_trimming(
        self,
        messages: List[Msg],
        prunable_indexes: List[int],
        usage_ratio: float,
    ) -> List[Msg]:
        """Apply soft-trimming to prunable messages.

        Args:
            messages (`List[Msg]`):
                The messages.
            prunable_indexes (`List[int]`):
                Indexes of prunable messages.
            usage_ratio (`float`):
                Current context usage ratio.

        Returns:
            `List[Msg]`:
                Messages with soft-trimming applied.
        """
        if usage_ratio < self.config.soft_trim_ratio:
            return messages

        result = list(messages)
        n_trimmed = 0

        for i in prunable_indexes:
            if usage_ratio < self.config.soft_trim_ratio:
                break

            msg = result[i]
            trimmed = soft_trim_tool_result(msg, self.config.soft_trim)

            if trimmed:
                result[i] = trimmed
                n_trimmed += 1

                # Recalculate usage
                total_chars = estimate_messages_total_chars(result)
                char_window = estimate_context_window_chars(200_000)
                usage_ratio = (
                    total_chars / char_window if char_window > 0 else 0
                )

        if n_trimmed > 0:
            logger.info(
                "Soft-trimmed %d tool result messages",
                n_trimmed,
            )

        return result

    def _apply_hard_clearing(
        self,
        messages: List[Msg],
        prunable_indexes: List[int],
        usage_ratio: float,
    ) -> List[Msg]:
        """Apply hard-clearing to prunable messages.

        Args:
            messages (`List[Msg]`):
                The messages.
            prunable_indexes (`List[int]`):
                Indexes of prunable messages.
            usage_ratio (`float`):
                Current context usage ratio.

        Returns:
            `List[Msg]`:
                Messages with hard-clearing applied.
        """
        if not self.config.hard_clear.enabled:
            return messages

        if usage_ratio < self.config.hard_clear_ratio:
            return messages

        result = list(messages)
        n_cleared = 0

        for i in prunable_indexes:
            if usage_ratio < self.config.hard_clear_ratio:
                break

            msg = result[i]
            cleared = hard_clear_tool_result(
                msg,
                self.config.hard_clear.placeholder,
            )
            result[i] = cleared
            n_cleared += 1

            # Recalculate usage
            total_chars = estimate_messages_total_chars(result)
            char_window = estimate_context_window_chars(200_000)
            usage_ratio = total_chars / char_window if char_window > 0 else 0

        if n_cleared > 0:
            logger.info(
                "Hard-cleared %d tool result messages",
                n_cleared,
            )

        return result
