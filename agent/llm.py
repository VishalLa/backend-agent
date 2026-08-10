from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from config import Config


OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/local-coding-agent",
    "X-Title": "Local Coding Agent",
}


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
        kwargs: dict[str, Any] = dict(
            model=self.config.ollama_model,
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


    def _fallback_chain(self, tools: Optional[list]) -> Any:
        """Build the runnable for this self.config, respecting self.config.provider."""
        if self.config.provider == "local":
            if not self.config.enable_ollama_fallback:
                raise ValueError(
                    "provider is 'local' but Ollama fallback is disabled "
                    "(enable_ollama_fallback=False). Enable it, or switch back "
                    "to provider='api'."
                )

            ollama = self._build_ollama()
            if ollama is None:
                raise ValueError(
                    "provider is 'local' but the Ollama client could not be built. "
                    "Check that Ollama is running and ollama_base_url is correct."
                )

            return ollama.bind_tools(tools) if tools else ollama

        primary = self._build_sambanova()
        openrouter = self._build_openrouter()
        groq = self._build_groq()
        ollama = self._build_ollama() if self.config.enable_ollama_fallback else None

        if tools:
            primary = primary.bind_tools(tools)
            if openrouter is not None:
                openrouter = openrouter.bind_tools(tools)
            if groq is not None:
                groq = groq.bind_tools(tools)
            if ollama is not None:
                ollama = ollama.bind_tools(tools)

        fallbacks = [m for m in (openrouter, groq, ollama) if m is not None]
        
        return primary.with_fallbacks(fallbacks, exceptions_to_handle=(Exception,)) if fallbacks else primary


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

