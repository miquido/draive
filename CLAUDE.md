# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup

- `make venv` - Setup development environment and install git hooks
- `source .venv/bin/activate && make sync` - Sync dependencies with uv lock file
- `source .venv/bin/activate && make update` - Update and lock dependencies

### Code Quality

- `source .venv/bin/activate && make format` - Format code with Ruff
- `source .venv/bin/activate && make lint` - Run linters (Ruff + Bandit + Pyright strict mode)
- `source .venv/bin/activate && make test` - Run pytest with coverage
- `source .venv/bin/activate && pytest tests/test_specific.py` - Run single test file
- `source .venv/bin/activate && pytest tests/test_specific.py::test_function` - Run specific test

## Architecture Overview

Draive is a Python framework (3.12+) for LLM applications built on the **Haiway** framework, leveraging its state, dependency, and task-management facilities. It emphasizes:

1. **Immutable State Management**: Type-safe, immutable data structures with validation
2. **Context-based Dependency Injection**: Safe state propagation in concurrent environments
3. **Functional Approach**: Pure functions over objects with methods
4. **Structured Concurrency**: Automatic task management and resource cleanup
5. **Haiway Proxy**: draive builds on top of haiway and exports its symbols

### Core Components

- **Integrations**: Various AI service provider integrations. Including Anthropic, AWS Bedrock, Cohere, Gemini, Mistral, Ollama, OpenAI, VLLM
- **Functionalities**: High- and low-level interfaces and implementations of application building blocks, including embedding, **LMM**, evaluation, generation, conversation, guardrails, instructions, resources, prompts, tools, **MCP** support
- **Parameters**: Immutable data structures and **parameterized** functions with **JSON Schema** generation, validation, and generic-type support
- **Utils**: Application utilities, common types and interfaces
- **Helpers**: Implementations of selected functionalities combining other components

### Code Style

- Use absolute imports from `draive` package
- Put exported symbols into `__init__.py`
- Follow Ruff import ordering (standard library, third party, local)
- Use Python 3.12+ type features (type unions with `|`, generic syntax)
- Use base and abstract types like `Sequence` or `Iterable` instead of concrete
- Use custom exceptions for specific errors

### Testing Guidelines

- Uses pytest with async support. Tests are in `tests/` directory.
- Mock dependencies within scope using stubbed functionality state.

## Examples

### Immutability Rules

**ALWAYS use these types for collections in State, Config and DataModel classes:**
- Use `Sequence[T]` instead of `list[T]` (becomes tuple)
- Use `Mapping[K,V]` instead of `dict[K,V]` (becomes immutable)
- Use `Set[T]` instead of `set[T]` (becomes frozenset)

```python
from typing import Sequence, Mapping, Set
from haiway import State

class UserData(State):
    roles: Sequence[str]  # Will be tuple
    metadata: Mapping[str, Any]  # Will be immutable
    tags: Set[str]  # Will be frozenset

```

### State Definition Patterns

```python
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4
from haiway import State
from draive import Field

# Basic data structure
class UserData(State):
    id: UUID
    name: str
    email: str | None = None

# Generic state classes
class Container[Element](State):
    items: Sequence[Element]
    metadata: Mapping[str, Any]

# Special kind of State supporting loading from external source
class UserConfig(Config):
    value: str

# Serializable counterpart of State
class UserModel(DataModel):
    # Field allows customizing DataModel fields with various options 
    id: UUID = Field(default_factory=uuid4, aliased="user_id")
    name: str

# Function protocol
@runtime_checkable
class UserFetching(Protocol):
    async def __call__(self, id: str) -> UserData: ...

# Functionality state pattern used for dependency injection
class UserService(State):
    # Function implementations
    user_fetching: UserFetching

    # Class method interface to access functions within context
    @classmethod
    async def fetch_user(cls, *, id: str) -> UserData:
        return await ctx.state(cls).user_fetching(id)
```

### State Updates

```python
# Immutable updates through copy, same for State, Config and DataModel
user: UserData = ...
updated_user: UserData = user.updated(name="Updated")
```

### Resource Management

```python
from contextlib import asynccontextmanager
from haiway import ctx, State

class ResourceAccess(State):
    accessing: ResourceAccessing

    @classmethod
    def access(cls) -> ResourceData:
        return ctx.state(cls).accessing()

@asynccontextmanager
async def create_resource_disposable():
    # Create a disposable resource
    resource: ResourceHandle = await open_resource()
    try:
        # Yield the state that will be made available in the context
        yield ResourceState(accessing=resource.access)

    finally:
        # Cleanup happens automatically when context exits
        await resource.close()

# Resources are automatically cleaned up and their state included in context
async with ctx.scope(
    "work",
    disposables=(create_resource_disposable(),)
):
    # ResourceAccess is now available in the context
    resource_data: ResourceData = ResourceAccess.access()
# Cleanup happens automatically here
```

## Testing Patterns

```python
import pytest
from haiway import ctx

@pytest.mark.asyncio
async def test_functionality():
    # Set up test context
    async with ctx.scope("test", TestState(value="test")):
        result = await some_function()
        assert result == expected_result

@pytest.mark.asyncio
async def test_with_mock():
    async with ctx.scope("test", ServiceState(fetching=mock_fetching)):
        # Test with mocked service
```
