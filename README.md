# Small Agent Code

A **proof of concept** demonstrating Anthropic's prompt caching with LangGraph agents to reduce API costs and improve response times.

## Overview

This PoC shows how to implement prompt caching in AI agents using Anthropic's Claude. The key innovation is in the `inject_cache_control` function (agent.py:64-90) that adds cache control metadata to messages, reducing costs by up to 90% for repeated prompts.

<img width="1979" height="1179" alt="Cumulative Cost over 20 Turns for Claude Sonnet 4 (With Cache Write on Turn 1)" src="https://github.com/user-attachments/assets/95158b0a-3d65-486e-8423-6afb8c31b4ac" />


## Key Implementation

The core caching logic injects `cache_control` metadata into messages:

```python
def inject_cache_control(state: Dict[str, Any]) -> Dict[str, List[Any]]:
    # Add ephemeral cache control to the latest message
    # System prompts get cached for reuse across conversations
    # This reduces token costs and improves response times
```

This function is registered as a `pre_model_hook` in the LangGraph agent to automatically handle caching.

## Setup

```bash
uv sync
```

Create `.env` with your Anthropic API key:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

```bash
uv run --env-file .env streamlit run app.py
```

## Files

- `agent.py` - Core agent with prompt caching implementation
- `app.py` - Streamlit web interface
- `main.py` - Simple CLI entry point

## Note

This is a proof of concept, not production-ready code. The caching implementation demonstrates the concept and can be adapted for production use cases.
