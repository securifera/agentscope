# -*- coding: utf-8 -*-
"""Pruning strategies for tool results."""

import fnmatch
from typing import Tuple, Optional, List

from ...message import Msg, TextBlock, ToolResultBlock
from ._config import (
    SoftTrimConfig,
    HardClearConfig,
    ToolPruningConfig,
)
from ._estimator import estimate_message_chars


def has_image_blocks(content: List[dict]) -> bool:
    """Check if content contains image blocks.

    Args:
        content (`List[dict]`):
            The content blocks to check.

    Returns:
        `bool`:
            True if content contains images.
    """
    if not isinstance(content, list):
        return False

    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            return True
    return False


def collect_text_segments(content: List[dict]) -> List[str]:
    """Collect all text segments from content blocks.

    Args:
        content (`List[dict]`):
            The content blocks.

    Returns:
        `List[str]`:
            List of text segments.
    """
    segments = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                segments.append(text)
    return segments


def estimate_joined_text_length(segments: List[str]) -> int:
    """Estimate the length of joined text segments.

    Args:
        segments (`List[str]`):
            The text segments.

    Returns:
        `int`:
            The estimated length including separators.
    """
    if not segments:
        return 0
    # Account for newline separators between segments
    return sum(len(s) for s in segments) + max(0, len(segments) - 1)


def take_head_from_joined_text(segments: List[str], max_chars: int) -> str:
    """Take the head portion of joined text segments.

    Args:
        segments (`List[str]`):
            The text segments.
        max_chars (`int`):
            Maximum characters to take.

    Returns:
        `str`:
            The head portion.
    """
    if max_chars <= 0 or not segments:
        return ""

    result = []
    remaining = max_chars

    for i, segment in enumerate(segments):
        if i > 0 and remaining > 0:
            result.append("\n")
            remaining -= 1
            if remaining <= 0:
                break

        if len(segment) <= remaining:
            result.append(segment)
            remaining -= len(segment)
        else:
            result.append(segment[:remaining])
            break

    return "".join(result)


def take_tail_from_joined_text(segments: List[str], max_chars: int) -> str:
    """Take the tail portion of joined text segments.

    Args:
        segments (`List[str]`):
            The text segments.
        max_chars (`int`):
            Maximum characters to take.

    Returns:
        `str`:
            The tail portion.
    """
    if max_chars <= 0 or not segments:
        return ""

    result = []
    remaining = max_chars

    for i in range(len(segments) - 1, -1, -1):
        segment = segments[i]

        if len(segment) <= remaining:
            result.insert(0, segment)
            remaining -= len(segment)
        else:
            result.insert(0, segment[-remaining:])
            break

        if i > 0 and remaining > 0:
            result.insert(0, "\n")
            remaining -= 1
            if remaining <= 0:
                break

    return "".join(result)


def soft_trim_tool_result(
    msg: Msg,
    config: SoftTrimConfig,
) -> Optional[Msg]:
    """Apply soft-trimming to a tool result message.

    Keeps head and tail of the content with "..." separator.
    Skips messages with image blocks.

    Args:
        msg (`Msg`):
            The tool result message.
        config (`SoftTrimConfig`):
            The soft-trim configuration.

    Returns:
        `Optional[Msg]`:
            The trimmed message, or None if no trimming applied.
    """
    # Find tool result blocks
    tool_results = msg.get_content_blocks("tool_result")
    if not tool_results:
        return None

    # Skip if contains images
    for result in tool_results:
        output = result.get("output", [])
        if isinstance(output, list) and has_image_blocks(output):
            return None

    # Collect text from all tool result blocks
    all_segments = []
    for result in tool_results:
        output = result.get("output", [])
        if isinstance(output, list):
            segments = collect_text_segments(output)
            all_segments.extend(segments)
        elif isinstance(output, str):
            all_segments.append(output)

    raw_len = estimate_joined_text_length(all_segments)
    if raw_len <= config.max_chars:
        return None

    head_chars = max(0, config.head_chars)
    tail_chars = max(0, config.tail_chars)

    if head_chars + tail_chars >= raw_len:
        return None

    # Extract head and tail
    head = take_head_from_joined_text(all_segments, head_chars)
    tail = take_tail_from_joined_text(all_segments, tail_chars)

    trimmed_text = f"{head}\n...\n{tail}"
    notice = (
        f"\n\n[Tool result trimmed: kept first {head_chars} chars "
        f"and last {tail_chars} chars of {raw_len} chars.]"
    )

    # Create new content with trimmed text
    new_content = []
    for block in msg.content:
        if isinstance(block, dict):
            if block.get("type") == "tool_result":
                new_content.append({
                    "type": "tool_result",
                    "id": block.get("id"),
                    "output": [{"type": "text", "text": trimmed_text + notice}],
                })
            else:
                new_content.append(block)
        else:
            new_content.append(block)

    return Msg(
        name=msg.name,
        content=new_content,
        role=msg.role,
    )


def hard_clear_tool_result(
    msg: Msg,
    placeholder: str,
) -> Msg:
    """Replace tool result content with a placeholder.

    Args:
        msg (`Msg`):
            The tool result message.
        placeholder (`str`):
            The placeholder text.

    Returns:
        `Msg`:
            The cleared message.
    """
    new_content = []
    for block in msg.content:
        if isinstance(block, dict):
            if block.get("type") == "tool_result":
                new_content.append({
                    "type": "tool_result",
                    "id": block.get("id"),
                    "output": [{"type": "text", "text": placeholder}],
                })
            else:
                new_content.append(block)
        else:
            new_content.append(block)

    return Msg(
        name=msg.name,
        content=new_content,
        role=msg.role,
    )


def is_tool_prunable(
    tool_name: str,
    config: ToolPruningConfig,
) -> bool:
    """Check if a tool result can be pruned based on allow/deny rules.

    Args:
        tool_name (`str`):
            The name of the tool.
        config (`ToolPruningConfig`):
            The tool pruning configuration.

    Returns:
        `bool`:
            True if the tool can be pruned.
    """
    tool_name_lower = tool_name.lower()

    # Check deny rules first (they override allow)
    for pattern in config.deny:
        if fnmatch.fnmatch(tool_name_lower, pattern.lower()):
            return False

    # Check allow rules
    if not config.allow:
        # Empty allow list means allow all
        return True

    for pattern in config.allow:
        if fnmatch.fnmatch(tool_name_lower, pattern.lower()):
            return True

    return False


def extract_tool_name_from_message(msg: Msg) -> Optional[str]:
    """Extract the tool name from a tool result message.

    Args:
        msg (`Msg`):
            The message.

    Returns:
        `Optional[str]`:
            The tool name, or None if not found.
    """
    # Look for tool result blocks
    for block in msg.content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            # The tool name might be in metadata or we might need to
            # look at the corresponding tool_use message
            # For now, return None as we need the tool_use context
            pass

    # If the message name contains tool info, extract it
    if msg.name and "tool" in msg.name.lower():
        return msg.name

    return None
