# First Steps


- **`State`**: For internal application state (can include non-serializable data like functions)
- **`DataModel`**: For serializable data with JSON schema support

```python
from draive import State, DataModel
from typing import Sequence, Mapping

# State for internal use
class AppConfig(State):
    api_key: str
    max_retries: int = 3
    timeout: float = 30.0

# DataModel for data exchange
class UserProfile(DataModel):
    name: str
    email: str
    preferences: Mapping[str, str]  # Becomes immutable dict
    tags: Sequence[str]  # Becomes tuple
```


```python
from draive import ctx
from haiway import LoggerObservability

# Create a context scope
async with ctx.scope(
    "my-app",  # Scope name for logging/metrics
    AppConfig(api_key="secret"),  # State objects
    observability=LoggerObservability(),  # Optional observability
):
    # Access state within scope
    config = ctx.state(AppConfig)
    print(f"Using timeout: {config.timeout}")

    # Update state locally
    with ctx.updated(config.updated(timeout=60.0)):
        # This scope has updated timeout
        new_config = ctx.state(AppConfig)


```python
from contextlib import asynccontextmanager
from draive import ctx
from draive.openai import OpenAI

@asynccontextmanager
async def create_database_pool():
    pool = await create_pool()
    try:
        yield DatabaseState(pool=pool)
    finally:
        await pool.close()

# Resources are automatically cleaned up
async with ctx.scope(
    "app",
    disposables=(
        OpenAI(),  # LLM client
        create_database_pool(),  # Custom resource
    ),
):
    # Use resources here


```python
from draive import tool, Argument
from typing import Literal

@tool(
    name="search_products",  # Optional: custom name for LLM
    description="Search for products in the catalog",  # Helps LLM understand usage
)
async def search_products(
    query: str,
    category: Literal["electronics", "books", "clothing"] | None = None,
    max_results: int = Argument(
        default=10,
        description="Maximum number of results to return"
    ),
) -> str:
    """
    Search for products in the catalog.

    Parameters
    ----------
    query : str
        Search query
    category : Literal["electronics", "books", "clothing"] | None
        Optional category filter
    max_results : int
        Maximum number of results to return

    Returns
    -------
    str
        Search results as formatted text
    """
    # Implementation here


```python
from draive import TextGeneration, Toolbox, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig

async with ctx.scope(
    "generation",
    disposables=(OpenAI(),),
    OpenAIResponsesConfig(model="gpt-4o-mini"),
):
    # Basic text generation
    result = await TextGeneration.generate(
        instructions="You are a helpful assistant",
        input="Explain quantum computing",
    )

    # Generation with tools
    result_with_tools = await TextGeneration.generate(
        instructions="You are a helpful shopping assistant",
        input="Find me electronics under $100",
        tools=Toolbox.of(
            search_products,
            suggest=search_products,  # Suggest this tool
        ),


```python
from draive import Conversation, ConversationMessage, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig

async with ctx.scope(
    "chat",
    disposables=(OpenAI(),),
    OpenAIResponsesConfig(model="gpt-4o-mini"),
):
    # Single completion
    response = await Conversation.completion(
        instructions="You are a helpful assistant",
        input="Tell me about machine learning",
        tools=[search_products],  # Available tools
    )



```python
from draive import MultimodalContent, TextGeneration, ResourceContent
from pathlib import Path

async def analyze_image(image_path: Path):
    # Load image data
    with open(image_path, "rb") as file:
        raw_bytes = file.read()

    # Create multimodal content
    content = MultimodalContent.of(
        "What's in this image? Describe in detail.",
        ResourceContent.of(raw_bytes, mime_type="image/jpeg"),
    )

    result = await TextGeneration.generate(
        instructions="You are an image analysis expert",
        input=content,
    )


```python
from draive import ModelGeneration, DataModel
from typing import Sequence, Literal

class Issue(DataModel):
    line: int
    description: str
    type: Literal["bug", "style", "performance", "security"]

class CodeReview(DataModel):
    summary: str
    issues: Sequence[Issue]
    suggestions: Sequence[str]
    severity: Literal["low", "medium", "high"]

async def review_code(code: str) -> CodeReview:
    return await ModelGeneration.generate(
        CodeReview,
        instructions="You are an expert code reviewer",
        input=f"Review this code:\n```python\n{code}\n```",


```python
from draive import split_text, ctx, VectorIndex
from draive.helpers import VolatileVectorIndex
from draive.openai import OpenAI, OpenAIEmbeddingConfig
from collections.abc import Sequence

class DocumentChunk(DataModel):
    full_document: str
    content: str

async def setup_rag_system(document: str):
    async with ctx.scope(
        "rag",
        # Prepare in-memory vector index
        VolatileVectorIndex(),
        OpenAIEmbeddingConfig(model="text-embedding-3-small"),
        disposables=(OpenAI(),),
    ):
        # Split document into chunks
        document_chunks = [
            DocumentChunk(
                full_document=document,
                content=chunk,
            )
            for chunk in split_text(
                text=document,
                separators=("\n\n", " "),
                part_size=64,
                part_overlap_size=16,
                count_size=len,
            )
        ]

        # Index the chunks
        await VectorIndex.index(
            DocumentChunk,
            values=document_chunks,
            attribute=DocumentChunk._.content,  # Use AttributePath
        )

        return ctx.state(VectorIndex)

@tool(name="search_documents")
async def search_documents(query: str) -> str:
    """Search through indexed documents."""
    results: Sequence[DocumentChunk] = await VectorIndex.search(
        DocumentChunk,
        query=query,
        limit=3,
    )


```python
from draive import Conversation, Toolbox, ctx
from draive.mcp import MCPClient
from draive.openai import OpenAI, OpenAIResponsesConfig

async with ctx.scope(
    "mcp_integration",
    OpenAIResponsesConfig(model="gpt-4o-mini"),
    disposables=(
        OpenAI(),
        MCPClient.stdio(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/path/to/directory",
            ],
        ),
    ),
):
    # Use MCP tools automatically
    response = await Conversation.completion(
        instructions="You can access files using available tools",
        input="What files are in the directory?",
        tools=await Toolbox.fetched(),  # Fetches MCP tools
    )

from typing import Protocol, runtime_checkable
from draive import State, ctx

@runtime_checkable
class UserRepository(Protocol):
    async def get_user(self, id: str) -> User: ...
    async def save_user(self, user: User) -> None: ...

class RepositoryState(State):
    user_repo: UserRepository

    @classmethod
    async def get_user(cls, user_id: str) -> User:
        repo = ctx.state(cls).user_repo

from draive.evaluation import evaluator, EvaluationScore, evaluator_scenario
from draive.evaluators import groundedness_evaluator, readability_evaluator

@evaluator(name="keyword_presence", threshold=0.8)
async def keyword_evaluator(
    content: str,
    /,
    required_keywords: list[str],
) -> EvaluationScore:
    found = sum(1 for keyword in required_keywords if keyword.lower() in content.lower())
    score = found / len(required_keywords) if required_keywords else 0

    return EvaluationScore.of(
        score,
        comment=f"Found {found}/{len(required_keywords)} keywords",
    )

@evaluator_scenario(name="content_quality")
async def content_quality_scenario(content: str, /, *, reference: str):
    from draive.evaluation import EvaluationScenarioResult

    return await EvaluationScenarioResult.evaluating(
        content,
        groundedness_evaluator.prepared(reference=reference),
        readability_evaluator.prepared(),
        keyword_evaluator.with_threshold(0.5).prepared(
            required_keywords=["AI", "technology"]
        ),

from draive import Conversation

async def stream_example():
    async with ctx.scope("stream", disposables=(OpenAI(),)):
        # Stream text generation using Conversation.completion
        stream = await Conversation.completion(
            instructions="Tell a story",
            input="Once upon a time...",
            stream=True,
        )

        full_text = ""
        async for chunk in stream:
            if chunk.content:
                text = str(chunk.content)
                print(text, end="", flush=True)
                full_text += text


from draive import ToolError, GenerationError

async def robust_generation():
    try:
        result = await TextGeneration.generate(
            instructions="You are helpful",
            input="Calculate the square root of -1",
            tools=[calculate],
        )
    except ToolError as e:
        ctx.log_warning(f"Tool failed: {e}")
        result = "I couldn't perform that calculation"
    except GenerationError as e:
        ctx.log_error(f"Generation failed: {e}")
        result = "I'm having trouble generating a response"


from draive import setup_logging, ctx
from haiway import LoggerObservability

# Setup logging
setup_logging("my-app")

async def observable_operation():
    async with ctx.scope(
        "operation",
        observability=LoggerObservability(level="DEBUG"),
    ):
        # All operations are automatically traced
        ctx.record({"operation": "start"})

        result = await TextGeneration.generate(
            instructions="You are helpful",
            input="Hello!",
        )

        ctx.record({"tokens_used": len(result.split())})

# Good: Use immutable collection types
class GoodState(State):
    users: Sequence[str]  # Becomes tuple
    settings: Mapping[str, Any]  # Becomes immutable dict
    tags: Set[str]  # Becomes frozenset

# Bad: Mutable collections
class BadState(State):
    users: list[str]  # Mutable - avoid!

# Always use context scopes for LLM operations
async def good_function():
    async with ctx.scope("app", disposables=(OpenAI(),)):
        result = await TextGeneration.generate(input="Hello")
        return result

# Bad: No context
def bad_function():

# Good: Proper type hints
@tool(description="Get current weather")
async def get_weather(location: str, units: str = "celsius") -> str:
    return f"Weather in {location}: 22°{units[0].upper()}"

# Bad: Missing type hints
@tool
async def bad_tool(location):  # No types, no docstring

# Problem: Calling context-dependent code outside scope
def bad_function():
    config = ctx.state(AppConfig)  # Error! No context

# Solution: Ensure context exists
async def good_function():
    async with ctx.scope("app", AppConfig()):

# Problem: Trying to mutate state
state = AppState(value=1)
state.value = 2  # Error! State is immutable

# Solution: Use updated()

# Problem: Manual resource management
client = OpenAI()
# ... use client
# Forgot to cleanup!

# Solution: Use disposables
async with ctx.scope("app", disposables=(OpenAI(),)):
    # Client is automatically cleaned up


1. Explore the Guides for specific use cases:
   - [Basic Usage](../guides/BasicUsage.md)
   - [Basic Tools Use](../guides/BasicToolsUse.md)
   - [Basic Conversation](../guides/BasicConversation.md)
   - [Basic Evaluation](../guides/BasicEvaluation.md)

2. Check out Cookbooks for complete examples:
   - [Basic RAG](../cookbooks/BasicRAG.md)
   - [Basic MCP](../cookbooks/BasicMCP.md)
   - [Basic Data Extraction](../cookbooks/BasicDataExtraction.md)

3. Learn about advanced topics:
   - [Advanced State](../guides/AdvancedState.md)
   - [Basic Stage Usage](../guides/BasicStageUsage.md)

Remember: Draive is designed for clarity and composability. Start simple, test often, and gradually add complexity as needed.
