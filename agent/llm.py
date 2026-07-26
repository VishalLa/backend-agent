from typing import Any, Optional

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import AgentConfig

_llm_cache: dict[tuple, Any] = {}
_llm_with_tools_cache: dict[tuple, Any] = {}

OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/local-coding-agent",
    "X-Title": "Local Coding Agent",
}


def _build_primary(config: AgentConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model_name,
        api_key=config.groq_api_key.get_secret_value(),
        base_url=config.groq_base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.request_timeout,
        max_retries=0,
    )


def _build_openrouter(config: AgentConfig) -> Optional[ChatOpenAI]:
    key = config.openrouter_key_str()
    if not key:
        return None
    return ChatOpenAI(
        model=config.fallback_model_name,
        api_key=key,
        base_url=config.openrouter_base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.request_timeout,
        max_retries=0,
        default_headers=OPENROUTER_HEADERS,
    )


def _build_ollama(config: AgentConfig) -> Optional[ChatOllama]:
    if not config.enable_ollama_fallback:
        return None
    kwargs: dict[str, Any] = dict(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        num_predict=config.ollama_num_predict,
        num_ctx=config.ollama_num_ctx,
        keep_alive=config.ollama_keep_alive,
    )
    if config.ollama_num_thread is not None:
        kwargs["num_thread"] = config.ollama_num_thread
    return ChatOllama(**kwargs)


def _fallback_chain(config: AgentConfig, tools: Optional[list]) -> Any:
    """Build the runnable for this config, respecting config.provider_mode:

    - "local": bypass the API chain entirely and talk to Ollama directly.
      Raises a clear error if Ollama fallback isn't enabled, rather than
      silently falling through to an API call the user explicitly opted out of.
    - "api" (default): the full chain — primary (Groq) -> OpenRouter (if an
      API key is configured) -> local Ollama Qwen2.5-Coder (if enabled) as
      the last resort if every API tier is unavailable or exhausted.

    Tools are bound to each tier individually BEFORE chaining fallbacks —
    bind_tools() then with_fallbacks(), not the reverse — so every tier that
    gets used can actually call tools.
    """
    if config.provider_mode == "local":
        ollama = _build_ollama(config)
        if ollama is None:
            raise ValueError(
                "provider_mode is 'local' but Ollama fallback is disabled "
                "(enable_ollama_fallback=False). Enable it, or switch back "
                "to provider_mode='api'."
            )
        return ollama.bind_tools(tools) if tools else ollama

    primary = _build_primary(config)
    openrouter = _build_openrouter(config)
    ollama = _build_ollama(config)

    if tools:
        primary = primary.bind_tools(tools)
        if openrouter is not None:
            openrouter = openrouter.bind_tools(tools)
        if ollama is not None:
            ollama = ollama.bind_tools(tools)

    fallbacks = [m for m in (openrouter, ollama) if m is not None]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def _cache_key(config: AgentConfig, extra: tuple = ()) -> tuple:
    return (
        config.provider_mode,
        config.model_name,
        config.fallback_model_name,
        config.ollama_model,
        config.enable_ollama_fallback,
        config.ollama_num_ctx,
        config.ollama_num_predict,
        config.ollama_keep_alive,
        config.ollama_num_thread,
        config.openrouter_key_str() != "",
        config.temperature,
        config.max_tokens,
        config.request_timeout,
    ) + extra


def get_llm(config: AgentConfig) -> Any:
    """Return a shared LLM runnable (no tools bound) with the full
    Groq -> OpenRouter -> local Ollama fallback chain."""
    cache_key = _cache_key(config)
    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = _fallback_chain(config, tools=None)
    return _llm_cache[cache_key]


def get_llm_with_tools(config: AgentConfig, tools: list) -> Any:
    """Return a shared LLM-with-tools-bound runnable (full fallback chain),
    creating it once per (config, toolset) combination."""
    tool_names = tuple(sorted(t.name for t in tools))
    cache_key = _cache_key(config, extra=(tool_names,))
    if cache_key not in _llm_with_tools_cache:
        _llm_with_tools_cache[cache_key] = _fallback_chain(config, tools=tools)
    return _llm_with_tools_cache[cache_key]


def reset_llm_cache() -> None:
    """Clear cached clients. Mainly useful in tests."""
    _llm_cache.clear()
    _llm_with_tools_cache.clear()
