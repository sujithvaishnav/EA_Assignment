# FastAPI Framework Tutorial and Best Practices

## Introduction to FastAPI
FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints. Built on top of Starlette for web tooling and Pydantic for data validation, FastAPI delivers speeds on par with NodeJS and Go.

## Key Features
- **Automatic Documentation**: Generates interactive Swagger UI (`/docs`) and ReDoc (`/redoc`) automatically using OpenAPI specifications.
- **Type Safety**: Leverages Python type hints for static analysis, autocompletion, and runtime input validation.
- **Asynchronous Concurrency**: Native support for `async def` and standard `def` path operation functions.

## Defining Routes and Parameters

### 1. Path Parameters
Path parameters are variable parts of a URL path enclosed in curly brackets:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id, "message": f"Fetched item {item_id}"}
```

### 2. Query Parameters
Function arguments that are not part of the path parameters are interpreted as URL query parameters:

```python
from typing import Optional

@app.get("/items/")
async def list_items(skip: int = 0, limit: int = 10, search: Optional[str] = None):
    return {"skip": skip, "limit": limit, "search": search}
```

### 3. Request Body with Pydantic
To declare a JSON request body, use a Pydantic model:

```python
from pydantic import BaseModel, Field

class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    price: float = Field(..., gt=0.0)
    tax: Optional[float] = None

@app.post("/items/", status_code=201)
async def create_item(item: ItemCreate):
    total_price = item.price + (item.tax or 0.0)
    return {"name": item.name, "total_price": total_price}
```

## Response Models and Status Codes
Use `response_model` in path decorators to filter output data, enforce schema consistency, and generate accurate OpenAPI documentation:

```python
class ItemResponse(BaseModel):
    name: str
    price: float

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    # Any sensitive fields or internal fields will be filtered out
    return {"name": "Widget", "price": 49.99, "internal_secret": "xyz123"}
```

## Dependency Injection System (`Depends`)
FastAPI features a powerful dependency injection system for code reuse, database session management, authentication, and security:

```python
from fastapi import Depends, HTTPException, status

def get_db_session():
    db = {"session_active": True}
    try:
        yield db
    finally:
        db["session_active"] = False

async def verify_token(token: str):
    if token != "secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return {"user": "admin"}

@app.get("/protected/data")
async def get_protected_data(
    user: dict = Depends(verify_token),
    db: dict = Depends(get_db_session)
):
    return {"user": user["user"], "db_status": db["session_active"]}
```

## Background Tasks and Middleware
FastAPI supports non-blocking background tasks and custom middleware:
- **BackgroundTasks**: Run tasks after returning the HTTP response (e.g., logging queries, sending emails).
- **CORS Middleware**: Enable cross-origin resource sharing for web frontends.
