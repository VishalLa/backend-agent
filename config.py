import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# api provider
DEFAULT_SAMBANOVA_MODAL = "gpt-oss-120b"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
# local
DEFAULT_OLLAMA_MODEL = "MFDoom/deepseek-coder-v2-tool-calling:16b"
DEFAULT_OLLAMA_KQUANT_MODEL = "deepseek-coder-v2:16b-lite-instruct-q4_K_M"
SUMMARY_MODAL = "phi4-mini-reasoning"

VALID_AGENT_TYPES = ("backend", "ml", "git", "algorithms")


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sambanova_api_key: Optional[SecretStr] = None
    sambanova_base_url: str = "https://api.sambanova.ai/v1"
    sambanova_model: str = DEFAULT_SAMBANOVA_MODAL

    groq_api_key: Optional[SecretStr] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = DEFAULT_GROQ_MODEL

    openrouter_api_key: Optional[SecretStr] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL

    enable_ollama_fallback: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_summary_model: str = SUMMARY_MODAL
    ollama_use_kquant: bool = False

    ollama_num_ctx: int = 8192
    ollama_num_predict: int = 4096
    ollama_keep_alive: str = "10m"
    ollama_flash_attention: bool = False
    ollama_kv_cache_type: str = "q8_0"
    ollama_num_thread: Optional[int] = None
    ollama_request_timeout: float = 300.0

    max_context_tokens: int = 8192

    confirm_all_tools: bool = False

    log_file: str = "agent_events.log"

    temperature: float = 0.1
    max_tokens: int = 4096
    max_iterations: int = 10
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    agent_type: str = "backend"
    provider: str = "local"

    # i am using local llm for my convension you can update the default model to api provider
    # per-task-type model overrides
    backend_model_name: str = DEFAULT_OLLAMA_MODEL
    ml_model_name: str = DEFAULT_OLLAMA_MODEL
    git_model_name: str = DEFAULT_OLLAMA_MODEL
    algo_model_name: str = DEFAULT_OLLAMA_MODEL

    postgres_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/data_agent"
    postgres_pool_size: int = 5
    postgres_echo: bool = False

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    celery_task_always_eager: bool = True


    @field_validator(
        "sambanova_api_key",
        "groq_api_key",
        "openrouter_api_key"
    )
    @classmethod
    def _key_not_empty(cls, v: Optional[SecretStr]) -> Optional[SecretStr]:
        if v is not None and not v.get_secret_value().strip():
            raise ValueError("api key is empty")
        return v


    @field_validator(
        "sambanova_model",
        "groq_model",
        "openrouter_model",
        "ollama_model",
        "ollama_summary_model",
        "backend_model_name",
        "ml_model_name",
        "git_model_name",
        "algo_model_name",
    )
    @classmethod
    def _model_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model name must not be empty")
        return v


    @field_validator("agent_type")
    @classmethod
    def _valid_agent_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_AGENT_TYPES:
            raise ValueError(f"agent_type must be one of {VALID_AGENT_TYPES}")
        return v


    @field_validator("provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("api", "local"):
            raise ValueError("provider must be 'api' or 'local'")
        return v


    @field_validator("temperature")
    @classmethod
    def _temp_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        return v


    @field_validator(
        "max_tokens",
        "max_iterations",
        "max_retries",
        "ollama_num_ctx",
        "ollama_num_predict",
        "max_context_tokens",
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v


    @field_validator(
        "request_timeout",
        "retry_backoff_seconds",
        "ollama_request_timeout"
    )
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be a positive number")
        return v


    @model_validator(mode="after")
    def _require_api_keys_for_api_provider(self) -> "Config":
        """provider='api' needs every fallback key; provider='local' needs none."""
        if self.provider != "api":
            return self

        missing = [
            env_name
            for env_name, key in (
                ("SAMBANOVA_API_KEY", self.sambanova_api_key),
                ("GROQ_API_KEY", self.groq_api_key),
                ("OPENROUTER_API_KEY", self.openrouter_api_key),
            )
            if key is None or not key.get_secret_value().strip()
        ]
        if missing:
            raise ValueError(
                "provider='api' requires these keys: " + ", ".join(missing) +
                ". Set them, or construct Config with provider='local' to use Ollama only."
            )
        return self


    def get_keys_str(self) -> tuple[str, str, str]:
        return (
            self.openrouter_api_key.get_secret_value().strip() if self.openrouter_api_key else "",
            self.groq_api_key.get_secret_value().strip() if self.groq_api_key else "",
            self.sambanova_api_key.get_secret_value() if self.sambanova_api_key else ""
        )


    def get_model_for_task(self, task_mode: str) -> str:
        from agent.task_profile import TASK_PROFILES

        profile = TASK_PROFILES.get(task_mode)

        if profile is None:
            return self.ollama_model

        field_name = profile.get("model_field")
        return getattr(self, field_name, self.ollama_model)


    def context_over_budget(self, current_tokens: int) -> bool:
        return current_tokens >= self.max_context_tokens


    @classmethod
    def from_env(cls, provider: Optional[str] = None) -> "Config":
        resolved_provider = (provider or os.environ.get("AGENT_PROVIDER", "local")).strip().lower()

        sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "")
        groq_key = os.environ.get("GROQ_API_KEY", "")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

        if resolved_provider == "api":
            required_keys = {
                "SAMBANOVA_API_KEY": sambanova_key,
                "GROQ_API_KEY": groq_key,
                "OPENROUTER_API_KEY": openrouter_key
            }

            for key_name, key_val in required_keys.items():
                if not key_val:
                    raise ValueError(f"{key_name} not found in environment.")

        enable_ollama = os.environ.get(
            "AGENT_ENABLE_OLLAMA_FALLBACK", "true"
        ).strip().lower() not in ("0", "false", "no", "off")

        confirm_all = os.environ.get(
            "AGENT_CONFIRM_ALL_TOOLS", "false"
        ).strip().lower() in ("1", "true", "yes", "on")

        ollama_use_kquant = os.environ.get(
            "OLLAMA_USE_KQUANT", "false"
        ).strip().lower() in ("1", "true", "yes", "on")

        ollama_flash_attention = os.environ.get(
            "OLLAMA_FLASH_ATTENTION", "false"
        ).strip().lower() in ("1", "true", "yes", "on")

        ollama_num_thread_value = os.environ.get("OLLAMA_NUM_THREADS")
        ollama_num_thread = int(ollama_num_thread_value) if ollama_num_thread_value and ollama_num_thread_value.strip() else None

        default_task_model = DEFAULT_OLLAMA_MODEL if resolved_provider == "local" else DEFAULT_SAMBANOVA_MODAL
        kquant_task_model = DEFAULT_OLLAMA_KQUANT_MODEL if resolved_provider == "local" else DEFAULT_SAMBANOVA_MODAL

        if resolved_provider == "local":
            # Use K-quant if enabled, else tool-calling model
            base_model = kquant_task_model if ollama_use_kquant else DEFAULT_OLLAMA_MODEL
            backend_model_name = base_model
            ml_model_name = base_model
            git_model_name = base_model
            algo_model_name = base_model
        else:
            backend_model_name = os.environ.get("BACKEND_MODEL_NAME", DEFAULT_SAMBANOVA_MODAL)
            ml_model_name = os.environ.get("ML_MODEL_NAME", DEFAULT_SAMBANOVA_MODAL)
            git_model_name = os.environ.get("GIT_MODEL_NAME", DEFAULT_GROQ_MODEL)
            algo_model_name = os.environ.get("ALGO_MODEL_NAME", DEFAULT_SAMBANOVA_MODAL)

        return cls(
            sambanova_api_key=SecretStr(sambanova_key) if sambanova_key else None,
            sambanova_base_url=os.environ.get("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1"),

            groq_api_key=SecretStr(groq_key) if groq_key else None,
            groq_base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),

            openrouter_api_key=SecretStr(openrouter_key) if openrouter_key else None,
            openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),

            enable_ollama_fallback=enable_ollama,
            confirm_all_tools=confirm_all,
            ollama_use_kquant=ollama_use_kquant,
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "10m"),
            ollama_flash_attention=ollama_flash_attention,
            ollama_kv_cache_type=os.environ.get("OLLAMA_KV_CACHE_TYPE", "q8_0"),
            ollama_num_thread=ollama_num_thread,

            provider=resolved_provider,
            agent_type=os.environ.get("AGENT_TYPE", "backend"),

            backend_model_name=backend_model_name,
            ml_model_name=ml_model_name,
            git_model_name=git_model_name,
            algo_model_name=algo_model_name,

            postgres_url=os.environ.get("POSTGRES_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/data_agent"),
            postgres_pool_size=int(os.environ.get("POSTGRES_POOL_SIZE", "5")),
            postgres_echo=os.environ.get("POSTGRES_ECHO", "false").strip().lower() in ("1", "true", "yes", "on"),

            celery_broker_url=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
            celery_result_backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
            celery_task_always_eager=os.environ.get("CELERY_TASK_ALWAYS_EAGER", "true").strip().lower() in ("1", "true", "yes", "on"),
        )
