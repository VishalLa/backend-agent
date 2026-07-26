from typing import Any

from langchain_openai import ChatOpenAI
from .config import AgentConfig

_llm_cache: dict[str, Any] = {}
_llm_with_tools_cache: dict[tuple, Any] = {}


def get_llm(config: AgentConfig) -> Any:
    """Return a shared LLM runnable for this config with OpenRouter fallback.

    Retries are handled explicitly in graph.py (so failures can be logged
    and surfaced with context) rather than inside the client, so max_retries
    is pinned to 0 on purpose.
    """
    
    groq_key = (
        config.groq_api_key.get_secret_value()
        if hasattr(config.groq_api_key, "get_secret_value")
        else str(config.groq_api_key)
    )

    openrouter_key = ""
    if hasattr(config, "openrouter_api_key") and config.openrouter_api_key:
        openrouter_key = (
            config.openrouter_api_key.get_secret_value()
            if hasattr(config.openrouter_api_key, "get_secret_value")
            else str(config.openrouter_api_key)
        )

    fallback_model_name = getattr(config, "fallback_model_name", "openai/gpt-oss-120b:free")
    openrouter_base_url = getattr(config, "openrouter_base_url", "https://openrouter.ai/api/v1")

    cache_key = f"{config.model_name}:{config.temperature}:{config.max_tokens}:{config.request_timeout}"

    if cache_key not in _llm_cache:
        primary_llm = ChatOpenAI(
            model=config.model_name,
            api_key=groq_key,
            base_url=getattr(config, "groq_base_url", "https://api.groq.com/openai/v1"),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.request_timeout,
            max_retries=config.max_retries
        )

        if openrouter_key:
            fallback_llm = ChatOpenAI(
                model=fallback_model_name, # Fixed: was model_name=fallback_model_name
                api_key=openrouter_key,
                base_url=openrouter_base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.request_timeout,
                max_retries=config.max_retries,
                default_headers={
                    "HTTP-Referer": "https://github.com/local-coding-agent",
                    "X-Title": "Local Coding Agent",
                }
            )
            _llm_cache[cache_key] = primary_llm.with_fallbacks([fallback_llm])
        else:
            _llm_cache[cache_key] = primary_llm

    return _llm_cache[cache_key]


def get_llm_with_tools(config: AgentConfig, tools: list) -> Any:
    """Return a shared LLM-with-tools-bound runnable, creating it once per
    (config, toolset) combination.
    
    If OpenRouter fallback is configured, tool bindings are attached to both
    the primary and fallback instances before combining them.
    """
    tool_names = tuple(sorted(t.name for t in tools))
    fallback_model = getattr(config, "fallback_model_name", "openai/gpt-oss-120b:free")
    cache_key = (config.model_name, fallback_model, config.temperature, config.max_tokens, tool_names)

    if cache_key not in _llm_with_tools_cache:
        groq_key = (
            config.groq_api_key.get_secret_value()
            if hasattr(config.groq_api_key, "get_secret_value")
            else str(config.groq_api_key)
        )

        openrouter_key = ""
        if hasattr(config, "openrouter_api_key") and config.openrouter_api_key:
            openrouter_key = (
                config.openrouter_api_key.get_secret_value()
                if hasattr(config.openrouter_api_key, "get_secret_value")
                else str(config.openrouter_api_key)
            )

        openrouter_base_url = getattr(config, "openrouter_base_url", "https://openrouter.ai/api/v1")

        primary_llm_bound = ChatOpenAI(
            model=config.model_name,
            api_key=groq_key,
            base_url=getattr(config, "groq_base_url", "https://api.groq.com/openai/v1"),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.request_timeout,
            max_retries=0,
        ).bind_tools(tools)

        if openrouter_key:
            fallback_llm_bound = ChatOpenAI(
                model=fallback_model,
                api_key=openrouter_key,
                base_url=openrouter_base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.request_timeout,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://github.com/local-coding-agent",
                    "X-Title": "Local Coding Agent",
                },
            ).bind_tools(tools)

            _llm_with_tools_cache[cache_key] = primary_llm_bound.with_fallbacks([fallback_llm_bound])
        else:
            _llm_with_tools_cache[cache_key] = primary_llm_bound

    return _llm_with_tools_cache[cache_key]


def reset_llm_cache() -> None:
    """Clear cached clients. Mainly useful in tests."""
    _llm_cache.clear()
    _llm_with_tools_cache.clear()
