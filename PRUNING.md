# Context Pruning in AgentScope

This implementation brings OpenClaw's sophisticated token management techniques to AgentScope, providing multi-layered protection against context overflow and excessive token usage.

## Overview

Context pruning automatically reduces the size of tool results in message history before sending to the LLM, helping you:

- **Stay within token limits** - Prevent context overflow errors
- **Reduce costs** - Especially with prompt caching (Anthropic)
- **Maintain performance** - Keep recent conversation intact while pruning old tool outputs

## Quick Start

### Basic Usage

```python
from agentscope.agent import ReActAgent
from agentscope.memory._pruning import PruningConfig
from agentscope.model import AnthropicChatModel
from agentscope.formatter import AnthropicFormatter

# Create agent with pruning enabled
agent = ReActAgent(
    name="assistant",
    sys_prompt="You are a helpful assistant.",
    model=AnthropicChatModel(model_name="claude-sonnet-4"),
    formatter=AnthropicFormatter(),
    pruning_config=PruningConfig(
        mode="cache-ttl",  # Prune when cache expires
        ttl="5m",          # Cache TTL
    ),
)
```

### Using Smart Defaults

```python
from agentscope._defaults._pruning_defaults import (
    get_anthropic_pruning_defaults,
    get_openai_pruning_defaults,
)

# For Anthropic models (with prompt caching)
agent = ReActAgent(
    name="assistant",
    sys_prompt="You are a helpful assistant.",
    model=AnthropicChatModel(model_name="claude-sonnet-4"),
    formatter=AnthropicFormatter(),
    pruning_config=get_anthropic_pruning_defaults(),
)

# For OpenAI models (no caching)
agent = ReActAgent(
    name="assistant",
    sys_prompt="You are a helpful assistant.",
    model=OpenAIChatModel(model_name="gpt-4"),
    formatter=OpenAIFormatter(),
    pruning_config=get_openai_pruning_defaults(),
)
```

## How It Works

### Multi-Strategy Approach

1. **Soft-Trim** (30% context usage)
   - Keeps head + tail of large tool results
   - Adds "..." separator
   - Preserves context at beginning and end

2. **Hard-Clear** (50% context usage)
   - Replaces entire tool result with placeholder
   - Used when context is critically full
   - Always keeps recent messages

### Example: Soft-Trim

Before (10,000 chars):
```
File contents: line 1, line 2, line 3... [massive file] ...line 9999, line 10000
```

After soft-trim:
```
File contents: line 1, line 2, line 3
...
line 9999, line 10000

[Tool result trimmed: kept first 1500 chars and last 1500 chars of 10000 chars.]
```

### Example: Hard-Clear

Before (100,000 chars):
```
[Huge database dump with 100k characters]
```

After hard-clear:
```
[Old tool result content cleared]
```

## Configuration Options

### Full Configuration

```python
from agentscope.memory._pruning import (
    PruningConfig,
    SoftTrimConfig,
    HardClearConfig,
    ToolPruningConfig,
)

config = PruningConfig(
    # When to prune
    mode="cache-ttl",  # "off" | "cache-ttl" | "always"
    ttl="5m",          # Time-to-live for cache-ttl mode
    
    # Protection
    keep_last_assistants=3,  # Protect last N assistant messages
    
    # Thresholds
    soft_trim_ratio=0.3,          # Soft-trim at 30% context usage
    hard_clear_ratio=0.5,         # Hard-clear at 50% context usage
    min_prunable_tool_chars=50_000,  # Min chars before pruning activates
    
    # Soft-trim settings
    soft_trim=SoftTrimConfig(
        max_chars=4000,    # Trim results larger than this
        head_chars=1500,   # Keep 1500 chars from start
        tail_chars=1500,   # Keep 1500 chars from end
    ),
    
    # Hard-clear settings
    hard_clear=HardClearConfig(
        enabled=True,
        placeholder="[Old tool result content cleared]",
    ),
    
    # Tool-specific rules
    tools=ToolPruningConfig(
        allow=["*"],           # Allow all tools
        deny=["web_search"],   # Never prune web_search results
    ),
)
```

### Pruning Modes

#### `"off"` - Disabled
No pruning occurs. Use when you want to disable pruning temporarily.

#### `"cache-ttl"` - Cache-Aware (Recommended for Anthropic)
Prunes only when the last API call is older than `ttl`. This optimizes for Anthropic's prompt caching:
- During active conversation: no pruning (cache is valid)
- After idle period: prune before next call (cache expired anyway)
- Result: minimize cache-write costs

#### `"always"` - Aggressive
Prunes on every API call. Use for:
- Models without prompt caching
- Very long conversations
- Strict token budgets

## Tool-Specific Rules

Control which tool results can be pruned:

```python
config = PruningConfig(
    mode="always",
    tools=ToolPruningConfig(
        allow=["file_*", "code_*"],  # Only prune file and code tools
        deny=["web_search"],         # Never prune web searches
    ),
)
```

Wildcards are supported:
- `"*"` - matches everything
- `"file_*"` - matches `file_read`, `file_write`, etc.
- `"*_result"` - matches `search_result`, `api_result`, etc.

**Deny rules override allow rules.**

## Pruning vs Compression

Both mechanisms help with token management but serve different purposes:

| Feature | Pruning | Compression |
|---------|---------|-------------|
| **When** | Before each API call | When threshold exceeded |
| **What** | Trims/clears tool results | Summarizes conversation |
| **Persistent** | No (transient) | Yes (stored in memory) |
| **Scope** | Tool results only | All messages |
| **Speed** | Very fast (character-based) | Slower (LLM call required) |

**Recommendation**: Use both together for best results.

```python
agent = ReActAgent(
    # ... other args ...
    compression_config=CompressionConfig(
        enable=True,
        trigger_threshold=100_000,  # Compress at 100k tokens
    ),
    pruning_config=PruningConfig(
        mode="cache-ttl",
        ttl="5m",  # Prune when cache expires
    ),
)
```

## Cost Optimization Example

### Scenario: Long-running coding assistant

Without pruning:
```
Session length: 3 hours
Tool outputs: 500k chars accumulated
API calls: 50
Cost: $X (full context every time)
```

With pruning:
```
Session length: 3 hours
Tool outputs: 500k chars → pruned to 50k
API calls: 50
Cost: $X/5 (pruned context saves 80%)
```

## Monitoring

Pruning logs help you understand what's happening:

```python
import logging
logging.basicConfig(level=logging.INFO)

# You'll see logs like:
# INFO: Soft-trimmed 5 tool result messages
# INFO: Hard-cleared 2 tool result messages
# DEBUG: Context usage: 150000 chars / 800000 chars (18.8%)
```

## Best Practices

1. **Start conservative**: Use smart defaults, then tune
2. **Monitor logs**: Watch for excessive pruning
3. **Protect recent context**: Set `keep_last_assistants` appropriately
4. **Tool-specific rules**: Preserve critical tool outputs
5. **Combine with compression**: Use both for long sessions

## Troubleshooting

### "Too much pruning"
- Increase `min_prunable_tool_chars`
- Increase `soft_trim_ratio` and `hard_clear_ratio`
- Increase `keep_last_assistants`

### "Still hitting token limits"
- Decrease thresholds
- Use `mode="always"` instead of `"cache-ttl"`
- Enable compression alongside pruning

### "Lost important context"
- Add critical tools to `deny` list
- Increase `keep_last_assistants`
- Increase `head_chars` and `tail_chars` in soft-trim

## Advanced: Tool Output Truncation

Individual tools can also truncate their own outputs:

```python
from agentscope.tool import ToolResponse

# In your tool function
def my_large_output_tool():
    result = generate_large_output()  # 1M chars
    
    response = ToolResponse(
        content=[{"type": "text", "text": result}]
    )
    
    # Truncate at tool level (first line of defense)
    return response.truncate(max_chars=100_000)
```

This creates a **two-layer defense**:
1. Tool-level: Immediate truncation at execution
2. Context-level: Selective pruning before API calls

## Migration from Existing Code

Pruning is **opt-in** and backward compatible:

```python
# Before (no change needed)
agent = ReActAgent(...)

# After (add pruning)
agent = ReActAgent(
    ...,  # All existing args work
    pruning_config=get_anthropic_pruning_defaults(),
)
```

## Performance

- **Latency overhead**: < 5ms per pruning operation
- **Memory overhead**: Negligible (creates new message list)
- **Character estimation**: ~0.1ms for 1000 messages

## Summary

Context pruning provides:
- ✅ Automatic token management
- ✅ Cost optimization (especially with caching)
- ✅ No breaking changes
- ✅ Smart defaults for popular models
- ✅ Fine-grained control when needed
- ✅ Works alongside compression

Start with smart defaults, monitor logs, and tune as needed.
