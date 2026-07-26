import os

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "openai/gpt-oss-120b"


class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    groq_api_key: SecretStr
    model_name: str = DEFAULT_MODEL
    temperature: float = 0.1
    max_tokens: int = 4096
    max_iterations: int = 7
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    @field_validator("groq_api_key")
    @classmethod
    def _key_not_empty(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("GROQ_API_KEY is empty. Set it in your environment or a .env file.")
        return v

    @field_validator("model_name")
    @classmethod
    def _model_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_name must not be empty")
        return v

    @field_validator("temperature")
    @classmethod
    def _temp_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("max_tokens", "max_iterations", "max_retries")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("request_timeout", "retry_backoff_seconds")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be a positive number")
        return v

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Build config from environment variables / .env. Raises a clear
        ValueError immediately if GROQ_API_KEY is missing, rather than
        letting a cryptic auth error surface later mid-conversation."""
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. Set it in your shell "
                "or a .env file (see .env.example)."
            )
        return cls(
            groq_api_key=SecretStr(api_key),
            model_name=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
            temperature=float(os.environ.get("GROQ_TEMPERATURE", 0.1)),
            max_tokens=int(os.environ.get("GROQ_MAX_TOKENS", 4096)),
            max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", 15)),
            request_timeout=float(os.environ.get("GROQ_REQUEST_TIMEOUT", 60.0)),
            max_retries=int(os.environ.get("GROQ_MAX_RETRIES", 3)),
        )
        