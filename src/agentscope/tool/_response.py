# -*- coding: utf-8 -*-
"""The tool response class."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .._utils._common import _get_timestamp
from ..message import AudioBlock, ImageBlock, TextBlock, VideoBlock


@dataclass
class ToolResponse:
    """The result chunk of a tool call."""

    content: List[TextBlock | ImageBlock | AudioBlock | VideoBlock]
    """The execution output of the tool function."""

    metadata: Optional[dict] = None
    """The metadata to be accessed within the agent, so that we don't need to
    parse the tool result block."""

    stream: bool = False
    """Whether the tool output is streamed."""

    is_last: bool = True
    """Whether this is the last response in a stream tool execution."""

    is_interrupted: bool = False
    """Whether the tool execution is interrupted."""

    id: str = field(default_factory=lambda: _get_timestamp(True))
    """The identity of the tool response."""

    truncated: bool = False
    """Whether the tool output was truncated."""

    original_length: Optional[int] = None
    """The original character length before truncation."""

    def truncate(
        self,
        max_chars: int,
        mode: str = "tail",
    ) -> "ToolResponse":
        """Truncate the tool response content to a maximum character limit.

        Args:
            max_chars (`int`):
                The maximum number of characters allowed.
            mode (`str`, defaults to `"tail"`):
                The truncation mode: "tail", "head", or "head-tail".

        Returns:
            `ToolResponse`:
                A new ToolResponse instance with truncated content.
        """
        from ._truncation import (
            truncate_tool_output,
            add_truncation_notice,
        )

        truncated_content, was_truncated, original_len = truncate_tool_output(
            self.content,
            max_chars,
            mode,
        )

        if not was_truncated:
            return self

        # Add truncation notice
        truncated_content = add_truncation_notice(
            truncated_content,
            original_len,
            max_chars,
        )

        return ToolResponse(
            content=truncated_content,
            metadata=self.metadata,
            stream=self.stream,
            is_last=self.is_last,
            is_interrupted=self.is_interrupted,
            id=self.id,
            truncated=True,
            original_length=original_len,
        )
