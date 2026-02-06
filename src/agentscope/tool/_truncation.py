# -*- coding: utf-8 -*-
"""Tool output truncation utilities."""

from typing import List, Tuple

from ..message import (
    TextBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock,
    ContentBlock,
)

# Default limits for tool outputs
DEFAULT_TOOL_OUTPUT_MAX_CHARS = 200_000
TOOL_ERROR_MAX_CHARS = 400
TOOL_RESULT_STREAM_MAX_CHARS = 8_000


def truncate_utf16_safe(text: str, max_chars: int) -> str:
    """Truncate text safely, avoiding breaking UTF-16 surrogate pairs.

    Args:
        text (`str`):
            The text to truncate.
        max_chars (`int`):
            The maximum number of characters.

    Returns:
        `str`:
            The truncated text.
    """
    if len(text) <= max_chars:
        return text

    # Truncate and ensure we don't break surrogate pairs
    truncated = text[:max_chars]
    # If the last character is a high surrogate, remove it
    if truncated and ord(truncated[-1]) in range(0xD800, 0xDC00):
        truncated = truncated[:-1]

    return truncated


def estimate_content_blocks_length(
    content: List[TextBlock | ImageBlock | AudioBlock | VideoBlock],
) -> int:
    """Estimate the total character length of content blocks.

    Args:
        content (`List[TextBlock | ImageBlock | AudioBlock | VideoBlock]`):
            The content blocks to estimate.

    Returns:
        `int`:
            The estimated character length.
    """
    total = 0
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                total += len(block.get("text", ""))
            elif block_type in ["image", "audio", "video"]:
                # Images, audio, and video have significant token cost
                # Use fixed estimates similar to OpenClaw
                total += 8_000
        elif isinstance(block, str):
            total += len(block)

    return total


def truncate_tool_output(
    content: List[TextBlock | ImageBlock | AudioBlock | VideoBlock],
    max_chars: int,
    mode: str = "tail",
) -> Tuple[
    List[TextBlock | ImageBlock | AudioBlock | VideoBlock],
    bool,
    int,
]:
    """Truncate tool output content to a maximum character limit.

    Args:
        content (`List[TextBlock | ImageBlock | AudioBlock | VideoBlock]`):
            The content blocks to truncate.
        max_chars (`int`):
            The maximum number of characters allowed.
        mode (`str`, defaults to `"tail"`):
            The truncation mode:
            - "tail": Keep the last max_chars characters
            - "head": Keep the first max_chars characters
            - "head-tail": Keep head and tail with separator

    Returns:
        `Tuple[List[ContentBlock], bool, int]`:
            A tuple of (truncated_content, was_truncated, original_length)
    """
    original_length = estimate_content_blocks_length(content)

    if original_length <= max_chars:
        return content, False, original_length

    # Extract all text from text blocks
    text_parts: List[str] = []
    non_text_blocks: List[ContentBlock] = []

    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            else:
                # Keep non-text blocks as-is (images, audio, video)
                non_text_blocks.append(block)
        elif isinstance(block, str):
            text_parts.append(block)

    if not text_parts:
        # No text to truncate, return as-is
        return content, False, original_length

    # Join all text
    full_text = "\n".join(text_parts)

    # Truncate based on mode
    if mode == "tail":
        truncated_text = (
            f"... (truncated) "
            f"{truncate_utf16_safe(full_text, max_chars)[-max_chars:]}"
        )
    elif mode == "head":
        truncated_text = (
            f"{truncate_utf16_safe(full_text, max_chars)[:max_chars]}"
            f"\n\n... (truncated)"
        )
    elif mode == "head-tail":
        # Split available chars between head and tail
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        head = truncate_utf16_safe(full_text, head_chars)[:head_chars]
        tail = truncate_utf16_safe(full_text, tail_chars)[-tail_chars:]
        truncated_text = f"{head}\n...\n{tail}"
    else:
        # Default to tail
        truncated_text = truncate_utf16_safe(full_text, max_chars)[-max_chars:]

    # Create new content with truncated text
    new_content: List[ContentBlock] = [
        {"type": "text", "text": truncated_text},
    ]

    # Add non-text blocks back (they don't count toward text limit)
    new_content.extend(non_text_blocks)

    return new_content, True, original_length


def add_truncation_notice(
    content: List[TextBlock | ImageBlock | AudioBlock | VideoBlock],
    original_length: int,
    truncated_length: int,
) -> List[TextBlock | ImageBlock | AudioBlock | VideoBlock]:
    """Add a truncation notice to the content.

    Args:
        content (`List[ContentBlock]`):
            The truncated content.
        original_length (`int`):
            The original character length.
        truncated_length (`int`):
            The truncated character length.

    Returns:
        `List[ContentBlock]`:
            The content with truncation notice appended.
    """
    notice = (
        f"\n\n[Tool output truncated: showing {truncated_length} of "
        f"{original_length} characters]"
    )

    # Add notice to the last text block, or create a new one
    result = list(content)
    for i in range(len(result) - 1, -1, -1):
        if isinstance(result[i], dict) and result[i].get("type") == "text":
            result[i] = {
                "type": "text",
                "text": result[i]["text"] + notice,
            }
            return result

    # No text block found, add a new one
    result.append({"type": "text", "text": notice})
    return result


def truncate_error_message(error_text: str) -> str:
    """Truncate error messages to a reasonable length.

    Args:
        error_text (`str`):
            The error message to truncate.

    Returns:
        `str`:
            The truncated error message (first line only).
    """
    trimmed = error_text.strip()
    if not trimmed:
        return ""

    # Get first line only
    first_line = trimmed.split("\n")[0].strip()
    if not first_line:
        return ""

    # Truncate if too long
    if len(first_line) > TOOL_ERROR_MAX_CHARS:
        return truncate_utf16_safe(first_line, TOOL_ERROR_MAX_CHARS) + "…"

    return first_line
