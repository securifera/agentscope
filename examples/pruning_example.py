#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Context Pruning with ReActAgent

This example demonstrates how to use context pruning to manage token usage
in long-running conversations with tool-heavy workloads.
"""

import asyncio
from agentscope.agent import ReActAgent
from agentscope.memory import InMemoryMemory
from agentscope.memory._pruning import PruningConfig
from agentscope.message import Msg
from agentscope._defaults._pruning_defaults import (
    get_anthropic_pruning_defaults,
    get_openai_pruning_defaults,
)


async def example_basic_pruning():
    """Basic pruning example with manual configuration."""
    print("=" * 60)
    print("Example 1: Basic Pruning Configuration")
    print("=" * 60)

    # Create pruning config
    pruning_config = PruningConfig(
        mode="cache-ttl",  # Prune when cache expires
        ttl="5m",          # 5 minute cache TTL
        keep_last_assistants=3,  # Protect last 3 assistant messages
    )

    print(f"Pruning mode: {pruning_config.mode}")
    print(f"Cache TTL: {pruning_config.ttl}")
    print(f"Protected messages: {pruning_config.keep_last_assistants}")
    print()


async def example_smart_defaults():
    """Example using smart defaults for different providers."""
    print("=" * 60)
    print("Example 2: Smart Defaults")
    print("=" * 60)

    # Anthropic (with caching)
    anthropic_config = get_anthropic_pruning_defaults()
    print(f"Anthropic defaults:")
    print(f"  Mode: {anthropic_config.mode}")
    print(f"  TTL: {anthropic_config.ttl}")
    print(f"  Soft-trim ratio: {anthropic_config.soft_trim_ratio}")
    print()

    # OpenAI (no caching)
    openai_config = get_openai_pruning_defaults()
    print(f"OpenAI defaults:")
    print(f"  Mode: {openai_config.mode}")
    print(f"  Keep last: {openai_config.keep_last_assistants}")
    print(f"  Soft-trim ratio: {openai_config.soft_trim_ratio}")
    print()


async def example_tool_specific_rules():
    """Example with tool-specific pruning rules."""
    print("=" * 60)
    print("Example 3: Tool-Specific Rules")
    print("=" * 60)

    from agentscope.memory._pruning import ToolPruningConfig

    # Create config with selective tool pruning
    pruning_config = PruningConfig(
        mode="always",
        tools=ToolPruningConfig(
            allow=["file_*", "execute_*"],  # Only prune file and exec tools
            deny=["file_read_critical"],    # Except critical reads
        ),
    )

    print(f"Allow patterns: {pruning_config.tools.allow}")
    print(f"Deny patterns: {pruning_config.tools.deny}")
    print()


async def example_combined_with_compression():
    """Example combining pruning with compression."""
    print("=" * 60)
    print("Example 4: Pruning + Compression")
    print("=" * 60)

    # NOTE: This is a conceptual example showing the configuration
    # You would need actual model and formatter instances to run this

    print("Using both pruning and compression:")
    print()
    print("Pruning (fast, transient):")
    print("  - Runs before each API call")
    print("  - Trims/clears tool results")
    print("  - Character-based estimation")
    print()
    print("Compression (slower, persistent):")
    print("  - Runs when threshold exceeded")
    print("  - Summarizes entire conversation")
    print("  - Requires LLM call")
    print()
    print("Together: Best of both worlds!")
    print()


async def example_monitoring_pruning():
    """Example showing how to monitor pruning activity."""
    print("=" * 60)
    print("Example 5: Monitoring Pruning")
    print("=" * 60)

    import logging

    # Enable INFO logging to see pruning activity
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("With INFO logging enabled, you'll see:")
    print()
    print("Example logs:")
    print("  INFO: Soft-trimmed 5 tool result messages")
    print("  INFO: Hard-cleared 2 tool result messages")
    print("  DEBUG: Context usage: 150000 chars / 800000 chars (18.8%)")
    print()


async def example_custom_soft_trim():
    """Example with customized soft-trim settings."""
    print("=" * 60)
    print("Example 6: Custom Soft-Trim Settings")
    print("=" * 60)

    from agentscope.memory._pruning import SoftTrimConfig

    pruning_config = PruningConfig(
        mode="always",
        soft_trim=SoftTrimConfig(
            max_chars=6000,    # Trim results larger than 6k chars
            head_chars=2500,   # Keep 2.5k from start
            tail_chars=2500,   # Keep 2.5k from end
        ),
    )

    print(f"Soft-trim configuration:")
    print(f"  Max chars: {pruning_config.soft_trim.max_chars}")
    print(f"  Head chars: {pruning_config.soft_trim.head_chars}")
    print(f"  Tail chars: {pruning_config.soft_trim.tail_chars}")
    print()
    print("Tool result transformation:")
    print("  Before: [10000 chars of output]")
    print("  After:  [first 2500 chars]")
    print("          ...")
    print("          [last 2500 chars]")
    print("          [Tool result trimmed: ...]")
    print()


async def example_tool_output_truncation():
    """Example showing tool-level output truncation."""
    print("=" * 60)
    print("Example 7: Tool-Level Truncation")
    print("=" * 60)

    from agentscope.tool import ToolResponse

    print("Two-layer defense:")
    print()
    print("Layer 1: Tool-level truncation (immediate)")
    print("  - Tool returns large output")
    print("  - Truncated at execution time")
    print("  - Prevents unbounded outputs")
    print()
    print("Example:")
    print("  response = ToolResponse(content=[...])")
    print("  return response.truncate(max_chars=100_000)")
    print()
    print("Layer 2: Context-level pruning (selective)")
    print("  - Prunes old tool results before API call")
    print("  - Keeps recent context intact")
    print("  - Optimizes token usage")
    print()


async def main():
    """Run all examples."""
    examples = [
        example_basic_pruning,
        example_smart_defaults,
        example_tool_specific_rules,
        example_combined_with_compression,
        example_monitoring_pruning,
        example_custom_soft_trim,
        example_tool_output_truncation,
    ]

    for example in examples:
        await example()
        await asyncio.sleep(0.1)  # Small delay between examples

    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Choose a pruning configuration that fits your use case")
    print("2. Enable INFO logging to monitor pruning activity")
    print("3. Tune settings based on your token usage patterns")
    print("4. Consider combining with compression for best results")
    print()
    print("For more details, see PRUNING.md")


if __name__ == "__main__":
    asyncio.run(main())
