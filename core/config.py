import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

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
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
SUMMARY_MODAL = "phi4-mini-reasoning"

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sambanova_api_key: SecretStr
    sambanova_base_url: str = "https://api.sambanova.ai/v1"
    sambanova_model: str = DEFAULT_SAMBANOVA_MODAL

    groq_api_key: SecretStr = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = DEFAULT_GROQ_MODEL

    openrouter_api_key: SecretStr = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL

    enable_ollama_fallback: bool = True
    ollama_base_url = "http://localhost:11434"
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_summary_model: str = SUMMARY_MODAL

    ollama_num_ctx: int = 8192 
    ollama_num_predict: int = 4096
    ollama_keep_alive: str = "10m"
    ollama_num_thread: Optional[int] = None
    ollama_request_timeout: float = 300.0

    confirm_all_tools: bool = False

    log_file: str = "agent_events.log"

    temperature: float = 0.1
    max_tokens: int = 4096
    max_iterations: int = 7
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    agent_type: str = "backend"
    provider: str = "api"

    @field_validator(
        "sambanova_api_key",
        "groq_api_key",
        "openrouter_api_key"
    )
    @classmethod
    def _key_not_empty(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError(f"{v} is empty")
        return v

    @field_validator(
        "sambanova_model",
        "groq_model",
        "openrouter_model",
        "ollama_model",
        "ollama_summary_model"
    )
    @classmethod
    def _model_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model name must not tbe empty")
        return v

    
    @field_validator("agent_type")
    @classmethod
    def _valid_agent_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("backend", "ml"):
            raise ValueError("agent_type must be 'backend' or 'ml'")
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

    
    @field_validator(
        "max_tokens", 
        "max_iterations", 
        "max_retries", 
        "ollama_num_ctx", 
        "ollama_num_predict"
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
    
    def get_keys_str(self) -> tuple[str, str, str]:
        return (
            self.openrouter_api_key.get_secret_value().strip() if self.openrouter_api_key else "",
            self.groq_api_key.get_secret_value().strip() if self.groq_api_key else "",
            self.sambanova_api_key.get_secret_value() if self.sambanova_api_key else ""
        )

    @classmethod
    def from_env(cls) -> "Config":
        sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "")
        groq_key = os.environ.get("GROQ_API_KEY", "")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

        required_keys = {
            "SAMBANOVA_API_KEY": sambanova_key,
            "GROQ_API_KEY": groq_key,
            "OPENROUTER_API_KEY": openrouter_key
        }

        for key_name, key_val in required_keys.items():
            if not key_val:
                raise ValueError(f"{key_name} not found in environment.")

        enable_ollama = os.environ.get(
            ("AGENT_ENABLE_OLLAMA_FALLBACK").strip().lower() not in ("0", "false", "no", "off")
        )
        confirm_all = os.environ.get(
            ("AGENT_CONFIRM_ALL_TOOLS", "false").strip().lower() in ("1", "true", "yes", "on")
        )

		return cls(
			sambanova_api_key=SecretStr(sambanova_key),
			sambanova_base_url=os.environ.get("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1"),

			groq_api_key=SecretStr(groq_key),
			groq_base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),

			openrouter_api_key=SecretStr(openrouter_key),
			openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),

			enable_ollama_fallback=enable_ollama,
			confirm_all_tools=confirm_all
		)