from __future__ import annotations

from pydantic import BaseModel, ConfigDict

class SchemaBase(BaseModel):
    """
    Base class for all Pydantic schemas.

    from_attributes=True allows schemas to be created directly
    from SQLAlchemy ORM objects.
    """

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=False,
        arbitrary_types_allowed=True,
    )