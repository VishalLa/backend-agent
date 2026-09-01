from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import httpx
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from config import Config
from log.log_event import log_event

from .context_window import ContextWindowHandler


OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/local-coding-agent",
    "X-Title": "Local Coding Agent",
}

REMOTE_PROVIDER_CONTEXT_FLOOR = 32_000


@dataclass
class ProviderEntry:
    name: str
    model: Any
    context_limit: int


class AllProvidersFailedError(RuntimeError):
    """Every configured provider failed. Carries the last exception seen so
    classify_error() upstream still has something concrete to classify,
    while individual per-provider failures are logged as they happen."""


class FallbackChatModel:
    """
    Explicit provider iteration, replacing langchain's opaque
    `.with_fallbacks()`.

    Two things `.with_fallbacks()` cannot do, which this does:
      1. Size the input differently per provider. A conversation that fits
         comfortably in SambaNova's context has no reason to be sent
         uncompressed to a local Ollama model capped at ollama_num_ctx -
         that guarantees silent server-side truncation (typically eating
         the system prompt, since it's first). Each provider gets the
         conversation compressed to fit ITS OWN context_limit before the
         call, not a one-size-fits-all payload sized for the biggest one.
      2. Report which provider actually failed and why, for every attempt -
         not just whichever happened to be last when everything failed.
    """

    def __init__(self, entries: List[ProviderEntry], config: Config):
        self._entries = [e for e in entries if e.model is not None]
        if not self._entries:
            raise ValueError("no LLM providers are configured/available")
        self._config = config

    def invoke(self, messages, **kwargs) -> Any:
        last_exc: Optional[Exception] = None

        for entry in self._entries:
            sized_messages = self._fit_to_limit(messages, entry.context_limit)
            try:
                response = entry.model.invoke(sized_messages, **kwargs)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, this is the fallback boundary
                last_exc = exc
                log_event(
                    self._config.log_file,
                    "provider_attempt_failed",
                    provider=entry.name,
                    error=str(exc)[:500],
                )
                continue

            if hasattr(response, "response_metadata"):
                response.response_metadata = {
                    **(response.response_metadata or {}),
                    "provider": entry.name,
                }
            return response

        raise AllProvidersFailedError(
            f"all {len(self._entries)} configured provider(s) failed; "
            f"last error ({self._entries[-1].name if self._entries else '?'}): {last_exc}"
        ) from last_exc


    def _fit_to_limit(self, messages, limit: int):
        window = ContextWindowHandler(
            self._config.model_copy(update={"max_context_tokens": limit})
        )
        if window.estimate_tokens(messages) <= limit:
            return messages

        result = window.prepare(messages, force_summarize=True)
        return result.messages

    def bind_tools(self, tools):  # noqa: D401 - intentionally a no-op passthrough
        return self


class ChatModel:
    def __init__(self, config: Config) -> None:
        self.config = config

        self._llm_cache: dict[tuple, Any] = {}
        self._llm_with_tools_cache: dict[tuple, Any] = {}


    def _get_secret(self, secret: Any) -> Optional[str]:
        """Safely extract the string from a Pydantic SecretStr to prevent validation errors."""
        if not secret:
            return None
        return secret.get_secret_value() if hasattr(secret, 'get_secret_value') else str(secret)


    def _build_sambanova(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.config.get_model_for_task(self.config.agent_type),
            api_key=self._get_secret(self.config.sambanova_api_key) or "missing_key",
            base_url=self.config.sambanova_base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout,
            max_retries=0
        )


    def _build_openrouter(self) -> Optional[ChatOpenAI]:
        api_key = self._get_secret(self.config.openrouter_api_key)
        if not api_key:
            return None

        try:
            return ChatOpenAI(
                model=self.config.openrouter_model,
                api_key=api_key,
                base_url=self.config.openrouter_base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
                default_headers=OPENROUTER_HEADERS,
                max_retries=0
            )
        except Exception as e:
            print(f"ERROR: failed to build OpenRouter client: {e}")
            return None


    def _build_groq(self) -> Optional[ChatOpenAI]:
        api_key = self._get_secret(self.config.groq_api_key)
        if not api_key:
            return None

        try:
            return ChatOpenAI(
                model=self.config.groq_model,
                api_key=api_key,
                base_url=self.config.groq_base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
                max_retries=0
            )
        except Exception as e:
            print(f"ERROR: failed to build Groq client: {e}")
            return None


    def _build_ollama(self) -> Optional[ChatOllama]:
        task_model = self.config.get_model_for_task(self.config.agent_type)
        kwargs: dict[str, Any] = dict(
            model=task_model,
            base_url=self.config.ollama_base_url,
            temperature=self.config.temperature,
            num_predict=self.config.ollama_num_predict,
            num_ctx=self.config.ollama_num_ctx,
            keep_alive=self.config.ollama_keep_alive,
        )
        if self.config.ollama_num_thread is not None:
            kwargs["num_thread"] = self.config.ollama_num_thread

        try:
            try:
                return ChatOllama(**kwargs, timeout=self.config.ollama_request_timeout)
            except Exception:
                return ChatOllama(**kwargs)
        except Exception as e:
            print(f"ERROR: failed to build Ollama client: {e}")
            return None


    def _ollama_context_limit(self) -> int:
        return max(1024, self.config.ollama_num_ctx - self.config.ollama_num_predict)


    def ollama_healthcheck(self, timeout: float = 3.0) -> dict:
        """Fast reachability + model-availability check for the configured
        Ollama host.


        Returns:
            {
                "reachable": bool,
                "error": str | None,
                "model_available": bool,
                "wanted_model": str,
                "available_models": list[str],
            }
        """
        base_url = self.config.ollama_base_url.rstrip("/")
        wanted = self.config.ollama_model

        try:
            resp = httpx.get(f"{base_url}/api/tags", timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a diagnostic, not a call path
            return {
                "reachable": False,
                "error": (
                    f"could not reach Ollama at {base_url}: {exc}. If this is a "
                    "remote host, check it's powered on, on the same network, and "
                    "started with OLLAMA_HOST=0.0.0.0:11434 (not just localhost)."
                ),
                "model_available": False,
                "wanted_model": wanted,
                "available_models": [],
            }

        try:
            available = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
        except Exception:
            available = []

        model_available = wanted in available
        return {
            "reachable": True,
            "error": None if model_available else (
                f"Ollama at {base_url} is reachable, but '{wanted}' is not pulled there. "
                f"Available: {available or '(none)'}. Run: ollama pull {wanted}"
            ),
            "model_available": model_available,
            "wanted_model": wanted,
            "available_models": available,
        }


    def _provider_entries(self, tools: Optional[list]) -> List[ProviderEntry]:
        def bind(model):
            if model is None:
                return None
            return model.bind_tools(tools) if tools else model

        if self.config.provider == "local":
            if not self.config.enable_ollama_fallback:
                raise ValueError(
                    "provider is 'local' but Ollama fallback is disabled "
                    "(enable_ollama_fallback=False). Enable it, or switch back "
                    "to provider='api'."
                )
            ollama = bind(self._build_ollama())
            if ollama is None:
                raise ValueError(
                    "provider is 'local' but the Ollama client could not be built. "
                    "Check that Ollama is running and ollama_base_url is correct."
                )
            return [ProviderEntry(
                "ollama", 
                ollama, 
                self._ollama_context_limit()
            )]

        entries = [
            ProviderEntry(
                "sambanova",
                bind(self._build_sambanova()),
                max(self.config.max_context_tokens, REMOTE_PROVIDER_CONTEXT_FLOOR),
            ),
            ProviderEntry(
                "openrouter",
                bind(self._build_openrouter()),
                max(self.config.max_context_tokens, REMOTE_PROVIDER_CONTEXT_FLOOR),
            ),
            ProviderEntry(
                "groq",
                bind(self._build_groq()),
                max(self.config.max_context_tokens, REMOTE_PROVIDER_CONTEXT_FLOOR),
            ),
        ]
        if self.config.enable_ollama_fallback:
            entries.append(
                ProviderEntry(
                    "ollama", 
                    bind(self._build_ollama()), 
                    self._ollama_context_limit()
                )
            )
        return entries


    def _fallback_chain(self, tools: Optional[list]) -> Any:
        """Build the runnable for this self.config, respecting self.config.provider.

        Returns a FallbackChatModel (see above) rather than a langchain
        RunnableWithFallbacks - same external .invoke() interface, but each
        provider gets input sized to its own real context capacity instead
        of one payload sized for the largest provider.
        """
        entries = self._provider_entries(tools)
        return FallbackChatModel(entries, self.config)


    def _cache_key(self, extra: tuple = ()) -> tuple:
        return (
            self.config.provider,
            self.config.agent_type,
            self.config.temperature
        ) + extra


    def get_llm(self) -> Any:
        cache_key = self._cache_key()
        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = self._fallback_chain(tools=None)
        return self._llm_cache[cache_key]


    def get_llm_with_tools(self, tools: list) -> Any:
        tool_names = tuple(sorted(t.name for t in tools))
        cache_key = self._cache_key(extra=(tool_names,))

        if cache_key not in self._llm_with_tools_cache:
            self._llm_with_tools_cache[cache_key] = self._fallback_chain(tools=tools)

        return self._llm_with_tools_cache[cache_key]


    def reset_llm_cache(self) -> None:
        self._llm_cache.clear()
        self._llm_with_tools_cache.clear()

