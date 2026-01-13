# Basic RAG

Build a simple retrieval-augmented generation flow using `VectorIndex` and tool calls.

This cookbook uses in-memory index storage (`VolatileVectorIndex`) to keep setup minimal while
keeping the same API shape as production backends.

## Setup

```python
from draive import load_env

load_env()
```

## Prepare Chunk Model And Chunks

Define a typed chunk model and split source text into retrieval units.

```python
from draive import State, split_text


document = """Draive uses typed state and scoped context for GenAI applications."""


class DocumentChunk(State, serializable=True):
    full_document: str
    content: str


chunks = [
    DocumentChunk(full_document=document, content=part)
    for part in split_text(
        text=document,
        separators=("\n\n", " "),
        part_size=64,
        part_overlap_size=16,
        count_size=len,
    )
]
```

## Build In-Memory Vector Index

`VectorIndex` is state-driven, so we provide both index implementation and embedding configuration in
`ctx.scope(...)`.

```python
from draive import VectorIndex, ctx
from draive.helpers import VolatileVectorIndex
from draive.openai import OpenAI, OpenAIEmbeddingConfig


index = VolatileVectorIndex()

async with ctx.scope(
    "indexing",
    index,
    OpenAIEmbeddingConfig(model="text-embedding-3-small"),
    disposables=(OpenAI(),),
):
    await VectorIndex.index(
        DocumentChunk,
        values=chunks,
        attribute=DocumentChunk._.content,
    )
```

## Use Retrieval Tool During Generation

Expose vector search as a tool. The model can call it to fetch relevant context before answering.

```python
from collections.abc import Sequence

from draive import TextGeneration, Toolbox, VectorIndex, ctx, tool
from draive.openai import OpenAI, OpenAIEmbeddingConfig, OpenAIResponsesConfig


@tool(name="search")
async def index_search_tool(query: str) -> str:
    results: Sequence[DocumentChunk] = await VectorIndex.search(
        DocumentChunk,
        query=query,
        limit=3,
    )
    return "\n---\n".join(item.content for item in results)


async with ctx.scope(
    "rag",
    index,
    OpenAIResponsesConfig(model="gpt-5-mini"),
    OpenAIEmbeddingConfig(model="text-embedding-3-small"),
    disposables=(OpenAI(),),
):
    answer: str = await TextGeneration.generate(
        instructions="Answer based on retrieved context.",
        input="What does Draive use to manage execution state?",
        # Suggest retrieval tool to increase first-turn usage likelihood.
        tools=Toolbox.of([index_search_tool], suggesting=index_search_tool),
    )

    print(answer)
```
