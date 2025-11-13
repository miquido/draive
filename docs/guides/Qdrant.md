# Qdrant integration

Draive layers a Qdrant-backed persistence surface (`draive.qdrant`) on top of the standard `haiway`
context machinery so that you can persist embeddings, run similarity queries, and keep configuration
alongside your workflows without wiring the SDK yourself.

## Bootstrapping the Qdrant context

Start by binding `QdrantClient` inside a `ctx.scope(...)`. The client lazily opens an `AsyncQdrantClient`
using the `QDRANT_HOST`/`QDRANT_PORT` environment variables (defaults: `localhost:6334`) and exposes
a `Qdrant` state that wraps all collection, storage, search, and delete helpers.

```python
from draive import ctx
from draive.embedding import Embedded
from draive.qdrant import Qdrant, QdrantClient
from draive.parameters import DataModel

class Document(DataModel):
    id: str
    text: str

async with ctx.scope(
    "qdrant-demo",
    QdrantClient(),
):
    await Qdrant.create_collection(Document, vector_size=1536)
    await Qdrant.store(
        Document,
        objects=(
            Embedded(value=Document(id="doc-1", text="hello"), vector=[0.1] * 1536),
        ),
    )
```

Access the bound `Qdrant` state anywhere inside the scope via `ctx.state(Qdrant)` or by calling the
`Qdrant` statemethods directly (they resolve to the active context automatically).

## Collections and indexes

Each data model maps to a dedicated Qdrant collection named after the `DataModel` class. Use
`Qdrant.create_collection(...)` to provision it with the correct vector size, optional datatype, and
metric (Cosine, Dot, Euclid, Manhattan). The `skip_existing=True` flag lets you rerun bootstrapping
scripts safely. When you need faster filtering on payload attributes, `Qdrant.create_index(...)` will
register a payload index of the requested schema type (`keyword`, `text`, `integer`, `float`, etc.)
against an `AttributePath` from your model class.

```python
await Qdrant.create_index(Document, path=Document._.text, index_type="text")
```

Use `Qdrant.collections()` to inspect what the active server exposes and `Qdrant.delete_collection(...)`
when you need to tear it down.

## Persisting and replaying records

`Qdrant.store(...)` expects an iterable of `Embedded[Model]` instances. Each record holds the typed
`value` (built with your `DataModel`) and the dense vector that should be stored in Qdrant. You can
batch writes with `batch_size`, retry failures, and parallelize uploads through `parallel_tasks`.

```python
await Qdrant.store(
    Document,
    objects=(Embedded(value=Document(id="doc-2", text="bye"), vector=[0.2] * 1536),),
)
```

Reading back content uses `Qdrant.fetch(...)`, which scrolls through the collection. You can
optionally supply a `AttributeRequirement` (from `haiway`) to translate into a Qdrant filter, control
the page size with `limit`, and keep pagination state via the returned `QdrantPaginationToken`. Set
`include_vector=True` to get the stored vector together with each record if you need to rerank,
re-embed, or audit the data.

```python
page = await Qdrant.fetch(Document, limit=10, include_vector=True)
for embedded in page.results:
    print(embedded.vector)
```

When the stored payload should be deleted use `Qdrant.delete(...)` with the same `AttributeRequirement`
shape that you used for filtering.

## Similarity search and vector results

`Qdrant.search(...)` wraps the low-level `AsyncQdrantClient.search` call. Provide a `query_vector` (just a
`Sequence[float]`) or `include_vector=True` to receive `QdrantResult` instances carrying `identifier`,
`score`, `vector`, and the typed `content`. Filtering/narrowing results again goes through
`AttributeRequirement`, while `score_threshold` and `limit` control how many candidates come back.

```python
results = await Qdrant.search(
    Document,
    query_vector=[0.3] * 1536,
    score_threshold=0.6,
    include_vector=True,
)
for result in results:
    print(result.score, result.content.text)
```

The helper will raise `QdrantException` if the underlying SDK reports problems and converts mixed
payload/vector IDs into UUIDs so you can correlate results across stores.

## High-level QdrantVectorIndex helper

`QdrantVectorIndex()` builds a Draive `VectorIndex` facade using `TextEmbedding` and `ImageEmbedding`.
It extracts text from strings, `TextContent`, or image `ResourceContent` (image-only) and embeds them
before storing. When searching, the helper turns strings, `TextContent`, or images back into vectors,
prepends optional reranking with `mmr_vector_similarity_search`, and hides the low-level
`QdrantResult` objects unless you ask for them explicitly.

```python
from draive.qdrant import QdrantVectorIndex

async with ctx.scope("qdrant-index", QdrantClient()):
    index = QdrantVectorIndex()
    await index.index(Document, values=[Document(id="doc-3", text="hello")], attribute=Document._.text)
    hits = await index.search(Document, query="hello", limit=5, rerank=True)
```

Use `index.delete(...)` to drop the stored embeddings for a given requirement set, reuse `AttributeRequirement`
logic for runtime filtering, and lean on the same `Qdrant` state for paging.

## Combining storage with workflows

Combine these helpers inside `ctx.scope(...)` to keep vector storage, configuration snapshots, and
retrievals aligned with the `haiway` lifecycle. The request-scoped state manages `AsyncQdrantClient`
connections, so you can spin up multiple contexts in the same process, reusing environment-variable
configuration as needed.
