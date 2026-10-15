# Pydantic V2 Data Validation and Settings Guide

## Overview of Pydantic V2
Pydantic is the most widely used data validation library for Python. In version 2 (V2), Pydantic was rewritten with its core engine (`pydantic-core`) in Rust, resulting in 5x to 50x performance improvements. Pydantic ensures that data conforms to type annotations and provides comprehensive error reporting.

## Defining Models with `BaseModel` and `Field`
Models are classes that inherit from `BaseModel`. Fields are defined using type hints, with optional constraints defined via `Field(...)`:

```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

class UserProfile(BaseModel):
    id: int = Field(..., description="Unique integer identifier")
    username: str = Field(..., min_length=3, max_length=30, pattern="^[a-zA-Z0-9_-]+$")
    email: EmailStr
    age: Optional[int] = Field(default=None, ge=18, le=120)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

## Validation in Pydantic V2

### 1. Field Validators (`@field_validator`)
In Pydantic V2, `@field_validator` replaces `@validator`. By default, it operates after standard Pydantic parsing (`mode='after'`), but can also run before (`mode='before'`):

```python
from pydantic import BaseModel, field_validator

class QueryPayload(BaseModel):
    query: str
    top_k: int = 5

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query string cannot be blank or whitespace only")
        return cleaned

    @field_validator("top_k")
    @classmethod
    def validate_top_k_range(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError("top_k must be between 1 and 20")
        return v
```

### 2. Model Validators (`@model_validator`)
Model validators inspect or transform the entire model, useful for cross-field consistency validation:

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode="after")
    def check_dates(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date cannot be later than end_date")
        return self
```

## Serialization and Export (`model_dump` and `model_dump_json`)
In Pydantic V2, legacy `.dict()` and `.json()` methods are replaced with:
- `model.model_dump()`: Returns a Python `dict`.
- `model.model_dump_json(indent=2)`: Returns a serialized JSON string.
- `exclude_unset=True`: Excludes fields that were not explicitly provided during initialization.
- `by_alias=True`: Uses field alias names if defined.

## Configuration with `model_config` and `ConfigDict`
Instead of an inner `class Config`, Pydantic V2 uses `model_config = ConfigDict(...)`:

```python
from pydantic import BaseModel, ConfigDict

class StrictConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",        # Disallow extra fields not in schema
        str_strip_whitespace=True, # Automatically trim leading/trailing whitespace
        populate_by_name=True  # Allow populating by field name or alias
    )
    title: str
    author: str
```
