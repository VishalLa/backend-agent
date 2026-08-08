import os
from typing import Optional

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass 


DEFAULT_SAMBANOVA = "gpt-oss-120b"
