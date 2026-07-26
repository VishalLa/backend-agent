from typing import Any

from langchain_groq import ChatGroq

from .config import AgentConfig

_llm_cache: dict[str, ChatGroq] = {}
_llm_with_tools_cache: dict[tuple, Any] = {}


def get_llm(config: AgentConfig) -> ChatGroq:
    """Return a shared ChatGroq client for this config, creating it once.

    Retries are handled explicitly in graph.py (so failures can be logged
    and surfaced with context) rather than inside the client, so max_retries
    is pinned to 0 here on purpose.
    """
    cache_key = f"{config.model_name}:{config.temperature}:{config.max_tokens}:{config.request_timeout}"
    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = ChatGroq(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.groq_api_key.get_secret_value(),
            timeout=config.request_timeout,
            max_retries=0,
        )
    return _llm_cache[cache_key]


def get_llm_with_tools(config: AgentConfig, tools: list) -> Any:
    """Return a shared LLM-with-tools-bound runnable, creating it once per
    (config, toolset) combination."""
    tool_names = tuple(sorted(t.name for t in tools))
    cache_key = (config.model_name, config.temperature, config.max_tokens, tool_names)
    if cache_key not in _llm_with_tools_cache:
        _llm_with_tools_cache[cache_key] = get_llm(config).bind_tools(tools)
    return _llm_with_tools_cache[cache_key]


def reset_llm_cache() -> None:
    """Clear cached clients. Mainly useful in tests."""
    _llm_cache.clear()
    _llm_with_tools_cache.clear()
    