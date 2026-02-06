# -*- coding: utf-8 -*-
"""Token and character estimation utilities for pruning."""

from typing import List

from ...message import Msg

# Character-to-token ratio estimate (conservative)
CHARS_PER_TOKEN_ESTIMATE = 4

# Estimated token cost for images and other media
IMAGE_CHAR_ESTIMATE = 8_000
AUDIO_CHAR_ESTIMATE = 4_000
VIDEO_CHAR_ESTIMATE = 10_000


def estimate_context_window_chars(context_window_tokens: int) -> int:
    """Convert a token-based context window to character estimate.

    Args:
        context_window_tokens (`int`):
            The context window size in tokens.

    Returns:
        `int`:
            The estimated character count.
    """
    return context_window_tokens * CHARS_PER_TOKEN_ESTIMATE


def estimate_message_chars(msg: Msg) -> int:
    """Estimate the character count for a message.

    Args:
        msg (`Msg`):
            The message to estimate.

    Returns:
        `int`:
            The estimated character count.
    """
    total = 0

    # Estimate role and metadata overhead
    total += len(msg.role) + 20  # Role + JSON overhead

    # Count content blocks
    for block in msg.content:
        if isinstance(block, dict):
            block_type = block.get("type")

            if block_type == "text":
                total += len(block.get("text", ""))
            elif block_type == "thinking":
                total += len(block.get("content", ""))
            elif block_type == "tool_use":
                # Tool use block overhead
                total += len(block.get("name", ""))
                # Estimate JSON args size
                args = block.get("arguments", {})
                if isinstance(args, dict):
                    total += sum(
                        len(str(k)) + len(str(v))
                        for k, v in args.items()
                    )
                total += 50  # JSON overhead
            elif block_type == "tool_result":
                # Tool result blocks handled separately
                output = block.get("output", "")
                if isinstance(output, str):
                    total += len(output)
                elif isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict):
                            item_type = item.get("type")
                            if item_type == "text":
                                total += len(item.get("text", ""))
                            elif item_type == "image":
                                total += IMAGE_CHAR_ESTIMATE
                            elif item_type == "audio":
                                total += AUDIO_CHAR_ESTIMATE
                            elif item_type == "video":
                                total += VIDEO_CHAR_ESTIMATE
                total += 50  # JSON overhead
            elif block_type == "image":
                total += IMAGE_CHAR_ESTIMATE
            elif block_type == "audio":
                total += AUDIO_CHAR_ESTIMATE
            elif block_type == "video":
                total += VIDEO_CHAR_ESTIMATE
        elif isinstance(block, str):
            total += len(block)

    return total


def estimate_messages_total_chars(messages: List[Msg]) -> int:
    """Estimate the total character count for a list of messages.

    Args:
        messages (`List[Msg]`):
            The messages to estimate.

    Returns:
        `int`:
            The total estimated character count.
    """
    return sum(estimate_message_chars(msg) for msg in messages)


def parse_duration_to_seconds(duration: str) -> float:
    """Parse a duration string to seconds.

    Supported formats: "5m", "1h", "30s", "2h30m"

    Args:
        duration (`str`):
            The duration string.

    Returns:
        `float`:
            The duration in seconds.

    Raises:
        `ValueError`:
            If the duration format is invalid.
    """
    duration = duration.strip().lower()
    if not duration:
        raise ValueError("Duration string cannot be empty")

    total_seconds = 0.0
    current_number = ""

    for char in duration:
        if char.isdigit() or char == ".":
            current_number += char
        elif char in ["s", "m", "h", "d"]:
            if not current_number:
                raise ValueError(
                    f"Invalid duration format: {duration}",
                )

            value = float(current_number)

            if char == "s":
                total_seconds += value
            elif char == "m":
                total_seconds += value * 60
            elif char == "h":
                total_seconds += value * 3600
            elif char == "d":
                total_seconds += value * 86400

            current_number = ""
        else:
            raise ValueError(
                f"Invalid character '{char}' in duration: {duration}",
            )

    if current_number:
        raise ValueError(
            f"Duration must end with a unit (s/m/h/d): {duration}",
        )

    return total_seconds
