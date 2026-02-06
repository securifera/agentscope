# -*- coding: utf-8 -*-
"""Tests for context pruning functionality."""

import pytest
from agentscope.memory._pruning import (
    PruningConfig,
    SoftTrimConfig,
    HardClearConfig,
    ToolPruningConfig,
    ContextPruner,
)
from agentscope.memory._pruning._estimator import (
    estimate_message_chars,
    parse_duration_to_seconds,
    estimate_context_window_chars,
)
from agentscope.memory._pruning._strategies import (
    is_tool_prunable,
    soft_trim_tool_result,
    hard_clear_tool_result,
)
from agentscope.message import Msg


class TestPruningConfig:
    """Test pruning configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PruningConfig()
        assert config.mode == "off"
        assert config.ttl == "5m"
        assert config.keep_last_assistants == 3
        assert config.soft_trim_ratio == 0.3
        assert config.hard_clear_ratio == 0.5

    def test_invalid_ratios(self):
        """Test validation of ratio values."""
        with pytest.raises(ValueError):
            PruningConfig(soft_trim_ratio=1.5)

        with pytest.raises(ValueError):
            PruningConfig(hard_clear_ratio=-0.1)

    def test_tool_pruning_config(self):
        """Test tool-specific pruning rules."""
        config = ToolPruningConfig(
            allow=["file_*"],
            deny=["file_delete"],
        )
        assert "file_*" in config.allow
        assert "file_delete" in config.deny


class TestDurationParsing:
    """Test duration string parsing."""

    def test_parse_seconds(self):
        """Test parsing seconds."""
        assert parse_duration_to_seconds("30s") == 30.0
        assert parse_duration_to_seconds("5s") == 5.0

    def test_parse_minutes(self):
        """Test parsing minutes."""
        assert parse_duration_to_seconds("5m") == 300.0
        assert parse_duration_to_seconds("1m") == 60.0

    def test_parse_hours(self):
        """Test parsing hours."""
        assert parse_duration_to_seconds("1h") == 3600.0
        assert parse_duration_to_seconds("2h") == 7200.0

    def test_parse_combined(self):
        """Test parsing combined durations."""
        assert parse_duration_to_seconds("1h30m") == 5400.0
        assert parse_duration_to_seconds("2h15m30s") == 8130.0

    def test_invalid_format(self):
        """Test invalid duration formats."""
        with pytest.raises(ValueError):
            parse_duration_to_seconds("invalid")

        with pytest.raises(ValueError):
            parse_duration_to_seconds("5")  # Missing unit


class TestCharEstimation:
    """Test character estimation for messages."""

    def test_estimate_text_message(self):
        """Test estimating text-only messages."""
        msg = Msg(
            name="user",
            content=[{"type": "text", "text": "Hello, world!"}],
            role="user",
        )
        chars = estimate_message_chars(msg)
        assert chars > 0
        assert chars >= len("Hello, world!")

    def test_estimate_tool_result(self):
        """Test estimating tool result messages."""
        msg = Msg(
            name="system",
            content=[
                {
                    "type": "tool_result",
                    "id": "call_123",
                    "output": [{"type": "text", "text": "x" * 1000}],
                }
            ],
            role="user",
        )
        chars = estimate_message_chars(msg)
        assert chars >= 1000

    def test_context_window_estimation(self):
        """Test converting tokens to characters."""
        chars = estimate_context_window_chars(100_000)
        assert chars == 400_000  # 4 chars per token


class TestToolPrunableCheck:
    """Test tool pruning rules."""

    def test_allow_all(self):
        """Test allowing all tools."""
        config = ToolPruningConfig(allow=["*"], deny=[])
        assert is_tool_prunable("any_tool", config) is True
        assert is_tool_prunable("file_read", config) is True

    def test_wildcard_patterns(self):
        """Test wildcard patterns."""
        config = ToolPruningConfig(allow=["file_*"], deny=[])
        assert is_tool_prunable("file_read", config) is True
        assert is_tool_prunable("file_write", config) is True
        assert is_tool_prunable("web_search", config) is False

    def test_deny_overrides_allow(self):
        """Test that deny rules override allow."""
        config = ToolPruningConfig(
            allow=["*"],
            deny=["file_delete"],
        )
        assert is_tool_prunable("file_read", config) is True
        assert is_tool_prunable("file_delete", config) is False


class TestSoftTrimming:
    """Test soft-trimming strategy."""

    def test_soft_trim_large_result(self):
        """Test soft-trimming a large tool result."""
        long_text = "x" * 10000
        msg = Msg(
            name="system",
            content=[
                {
                    "type": "tool_result",
                    "id": "call_123",
                    "output": [{"type": "text", "text": long_text}],
                }
            ],
            role="user",
        )

        config = SoftTrimConfig(
            max_chars=4000,
            head_chars=1500,
            tail_chars=1500,
        )

        trimmed = soft_trim_tool_result(msg, config)
        assert trimmed is not None

        # Check that result is trimmed
        tool_result = trimmed.get_content_blocks("tool_result")[0]
        output_text = tool_result["output"][0]["text"]
        assert "..." in output_text
        assert "[Tool result trimmed:" in output_text

    def test_no_trim_small_result(self):
        """Test that small results are not trimmed."""
        small_text = "x" * 100
        msg = Msg(
            name="system",
            content=[
                {
                    "type": "tool_result",
                    "id": "call_123",
                    "output": [{"type": "text", "text": small_text}],
                }
            ],
            role="user",
        )

        config = SoftTrimConfig(max_chars=4000)
        trimmed = soft_trim_tool_result(msg, config)
        assert trimmed is None  # No trimming needed


class TestHardClearing:
    """Test hard-clearing strategy."""

    def test_hard_clear_result(self):
        """Test hard-clearing a tool result."""
        msg = Msg(
            name="system",
            content=[
                {
                    "type": "tool_result",
                    "id": "call_123",
                    "output": [{"type": "text", "text": "x" * 100000}],
                }
            ],
            role="user",
        )

        cleared = hard_clear_tool_result(msg, "[Cleared]")

        tool_result = cleared.get_content_blocks("tool_result")[0]
        output_text = tool_result["output"][0]["text"]
        assert output_text == "[Cleared]"


@pytest.mark.asyncio
class TestContextPruner:
    """Test the main context pruner."""

    async def test_pruner_off_mode(self):
        """Test that off mode doesn't prune."""
        config = PruningConfig(mode="off")
        pruner = ContextPruner(config)

        messages = [
            Msg(name="user", content="Hello", role="user"),
            Msg(name="assistant", content="Hi", role="assistant"),
        ]

        result = await pruner.prune_messages(messages)
        assert result == messages  # No changes

    async def test_should_prune_cache_ttl(self):
        """Test cache-ttl mode pruning logic."""
        import time

        config = PruningConfig(mode="cache-ttl", ttl="1s")
        pruner = ContextPruner(config)

        # Should not prune on first call (no last_api_call_time)
        assert pruner.should_prune(None) is False

        # Should not prune immediately after call
        assert pruner.should_prune(time.time()) is False

        # Should prune after TTL expires
        past_time = time.time() - 2  # 2 seconds ago
        assert pruner.should_prune(past_time) is True

    async def test_should_prune_always_mode(self):
        """Test always mode always returns True."""
        config = PruningConfig(mode="always")
        pruner = ContextPruner(config)

        assert pruner.should_prune(None) is True
        assert pruner.should_prune(0) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
