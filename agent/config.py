import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Provider-specific default model strings
DEFAULT_SAMBANOVA_BACKEND = "Meta-Llama-3.3-70B-Instruct"
DEFAULT_SAMBANOVA_ML = "gpt-oss-120b"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:32b"


class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- primary: SambaNova ---
    sambanova_api_key: SecretStr
    sambanova_base_url: str = "https://api.sambanova.ai/v1"
    model_name: str = DEFAULT_SAMBANOVA_BACKEND

    # --- task-based primary model selection ---
    task_mode: str = "backend"
    backend_model_name: str = DEFAULT_SAMBANOVA_BACKEND
    ml_model_name: str = DEFAULT_SAMBANOVA_ML

    # --- fallback tier 1: OpenRouter (different provider/quota) ---
    openrouter_api_key: Optional[SecretStr] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    fallback_model_name: str = DEFAULT_OPENROUTER_MODEL

    # --- fallback tier 2: Groq ---
    groq_api_key: Optional[SecretStr] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_name: str = DEFAULT_GROQ_MODEL

    # --- fallback tier 3: local Ollama (Qwen2.5-Coder) ---
    enable_ollama_fallback: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_num_ctx: int = 8192 
    ollama_num_predict: int = 4096
    ollama_keep_alive: str = "5m"
    ollama_num_thread: Optional[int] = None
    ollama_request_timeout: float = 300.0

    provider_mode: str = "api"

    confirm_all_tools: bool = False

    log_file: str = "agent_events.log"

    temperature: float = 0.1
    max_tokens: int = 4096
    max_iterations: int = 7
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    @field_validator("sambanova_api_key")
    @classmethod
    def _key_not_empty(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("SAMBANOVA_API_KEY is empty. Set it in your environment or a .env file.")
        return v

    @field_validator("model_name", "fallback_model_name", "groq_model_name", "ollama_model", "backend_model_name", "ml_model_name")
    @classmethod
    def _model_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model name must not be empty")
        return v

    @field_validator("task_mode")
    @classmethod
    def _valid_task_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("backend", "ml"):
            raise ValueError("task_mode must be 'backend' or 'ml'")
        return v

    @field_validator("provider_mode")
    @classmethod
    def _valid_provider_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("api", "local"):
            raise ValueError("provider_mode must be 'api' or 'local'")
        return v

    @field_validator("temperature")
    @classmethod
    def _temp_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("max_tokens", "max_iterations", "max_retries", "ollama_num_ctx", "ollama_num_predict")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("request_timeout", "retry_backoff_seconds", "ollama_request_timeout")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be a positive number")
        return v

    def openrouter_key_str(self) -> str:
        return self.openrouter_api_key.get_secret_value().strip() if self.openrouter_api_key else ""
        
    def groq_key_str(self) -> str:
        return self.groq_api_key.get_secret_value().strip() if self.groq_api_key else ""

    @classmethod
    def from_env(cls) -> "AgentConfig":
        sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "")
        if not sambanova_key:
            raise ValueError(
                "SAMBANOVA_API_KEY not found in environment. Set it in your shell "
                "or a .env file (see .env.example)."
            )

        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        groq_key = os.environ.get("GROQ_API_KEY", "").strip()
        
        enable_ollama = os.environ.get("AGENT_ENABLE_OLLAMA_FALLBACK", "true").strip().lower() not in (
            "0", "false", "no", "off",
        )
        confirm_all = os.environ.get("AGENT_CONFIRM_ALL_TOOLS", "false").strip().lower() in (
            "1", "true", "yes", "on",
        )

        task_mode = os.environ.get("AGENT_TASK_MODE", "backend").strip().lower()
        backend_model = os.environ.get("BACKEND_MODEL", DEFAULT_SAMBANOVA_BACKEND)
        ml_model = os.environ.get("ML_MODEL", DEFAULT_SAMBANOVA_ML)
        explicit_model = os.environ.get("SAMBANOVA_MODEL")
        
        # Primary model logic
        model_name = explicit_model or (backend_model if task_mode == "backend" else ml_model)

        num_thread_env = os.environ.get("OLLAMA_NUM_THREAD", "").strip()

        return cls(
            sambanova_api_key=SecretStr(sambanova_key),
            sambanova_base_url=os.environ.get("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1"),
            model_name=model_name,
            task_mode=task_mode,
            backend_model_name=backend_model,
            ml_model_name=ml_model,
            openrouter_api_key=SecretStr(openrouter_key) if openrouter_key else None,
            openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            fallback_model_name=os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
            groq_api_key=SecretStr(groq_key) if groq_key else None,
            groq_base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            groq_model_name=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            enable_ollama_fallback=enable_ollama,
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            ollama_num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", 8192)),
            ollama_num_predict=int(os.environ.get("OLLAMA_NUM_PREDICT", 4096)),
            ollama_keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "5m"),
            ollama_num_thread=int(num_thread_env) if num_thread_env else None,
            ollama_request_timeout=float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", 300.0)),
            provider_mode=os.environ.get("AGENT_PROVIDER_MODE", "api"),
            confirm_all_tools=confirm_all,
            log_file=os.environ.get("AGENT_LOG_FILE", "agent_events.log"),
            temperature=float(os.environ.get("AGENT_TEMPERATURE", 0.1)),
            max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", 4096)),
            max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", 20)),
            request_timeout=float(os.environ.get("AGENT_REQUEST_TIMEOUT", 60.0)),
            max_retries=int(os.environ.get("AGENT_MAX_RETRIES", 3)),
        )
