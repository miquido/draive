# Qdrant integration

Draive layers a Qdrant-backed persistence surface (`draive.qdrant`) on top of the standard `haiway`
context machinery so that you can persist embeddings, run similarity queries, and keep configuration
alongside your workflows without wiring the SDK yourself.

## Bootstrapping the Qdrant context

Start by binding `QdrantClient` inside a `ctx.scope(...)`. The client lazily opens an
`AsyncQdrantClient` using the `QDRANT_HOST`/`QDRANT_PORT` environment variables (defaults:
`localhost:6333`, with gRPC on `QDRANT_GRPC_PORT`, default `6334`) and exposes a `Qdrant` state
that wraps all collection, storage, search, and delete helpers.

```python
from draive import ctx, Pagination, State
from draive.embedding import Embedded
from draive.qdrant import Qdrant, QdrantClient

class Document(State, serializable=True):
    id: str
    text: str

async with ctx.scope(
    "qdrant-demo",
    disposables=[QdrantClient()],
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

Each data model maps to a dedicated Qdrant collection named after the `State` class. Use
`Qdrant.create_collection(...)` to provision it with the correct `vector_size`, optional
`vector_type` (`float32`, `float16`, `uint8`), and `distance` metric (`Cosine`, `Dot`, `Euclid`,
`Manhattan`). The `skip_existing=True` flag lets you rerun bootstrapping scripts safely. When you
need faster filtering on payload attributes, `Qdrant.create_index(...)` will register a payload
index of the requested schema type (`keyword`, `text`, `integer`, `float`, etc.) against an
`AttributePath` from your model class.

```python
await Qdrant.create_index(Document, path=Document._.text, index_type="text")
```

Use `Qdrant.collections()` to inspect what the active server exposes and
`Qdrant.delete_collection(...)` when you need to tear it down.

## Persisting and replaying records

`Qdrant.store(...)` expects an iterable of `Embedded[Model]` instances. Each record holds the typed
`value` (built with your `State`) and the dense vector that should be stored in Qdrant. You can
batch writes with `batch_size`, retry failures with `max_retries`, and parallelize uploads through
`parallel_tasks`.

```python
await Qdrant.store(
    Document,
    objects=(Embedded(value=Document(id="doc-2", text="bye"), vector=[0.2] * 1536),),
)
```

Reading back content uses `Qdrant.fetch(...)`, which scrolls through the collection. You can
optionally supply a `AttributeRequirement` (from `haiway`) to translate into a Qdrant filter and
control the page size with `Pagination.of(limit=...)`. The call returns a `Paginated`, so feed its
`pagination` back into the next `fetch(...)` to continue scrolling. Set `include_vector=True` to get
the stored vector together with each record if you need to rerank, re-embed, or audit the data.

```python
page = await Qdrant.fetch(
    Document,
    pagination=Pagination.of(limit=10),
    include_vector=True,
)
for embedded in page.items:
    print(embedded.vector)
```

When the stored payload should be deleted use `Qdrant.delete(...)` with the same
`AttributeRequirement` shape that you used for filtering.

### Requirement translation

`AttributeRequirement` operators map onto Qdrant filters as follows:

- `equal` / `not_equal` become a `must` / `must_not` field condition. Booleans and strings match
    through `MatchValue`, UUIDs match their string form, while numbers and datetimes match through a
    degenerate `Range` / `DatetimeRange`. `MatchValue` matches an integer only against an integer
    payload, so a payload holding `2.0` would not be matched by `2` - a range compares numbers by
    value, and payload datetimes likewise compare by value only as a range.
- `contained_in` and `contains_any` collapse into a single `MatchAny` when every value matches
    exactly, and fall back to a `should` disjunction when any of them needs a range condition -
    which is the case for every collection of numbers.
- `contains` matches an array payload field against a single value, which Qdrant satisfies when any
    element equals it.
- `text_match` becomes a `MatchText` condition - pair it with a `text` payload index.
- `and` / `or` nest as `must` / `should` filters.

Any other value type raises `NotImplementedError` instead of being silently dropped.

### Extra arguments

`create_collection`, `create_index`, `store`, `fetch`, `search` and `delete` accept `**extra`
forwarded to the underlying `AsyncQdrantClient` call (`create_collection`, `create_payload_index`,
`upload_points`, `scroll`, `query_points`, `delete`). Names the client does not declare raise a
`ValueError` listing them, since the client would otherwise assert on them - or drop them silently
when running with assertions disabled.

## Similarity search and vector results

`Qdrant.search(...)` wraps the low-level `AsyncQdrantClient.query_points` call. It takes a
`query_vector` (just a `Sequence[float]`) and returns the typed models by default; pass
`include_vector=True` to receive `QdrantResult` instances carrying `identifier`, `score`, `vector`,
and the typed `content`. Filtering/narrowing results again goes through `AttributeRequirement`,
while `score_threshold` and `limit` control how many candidates come back.

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

Point identifiers are normalized into UUIDs - both the integer and string forms Qdrant may report -
so you can correlate results across stores.

### Error translation

Every operation translates client failures into `QdrantException`, naming the operation and the
collection it was executed for and keeping the original SDK error as its `__cause__`. The client
communicates over gRPC by default, so untranslated failures would otherwise surface as `grpc`
errors carrying no indication of what failed. Argument verification is not a client failure - an
unsupported `**extra` name still raises `ValueError`.

## High-level QdrantVectorIndex helper

`QdrantVectorIndex.prepare()` builds a Draive `VectorIndex` facade using `TextEmbedding` and
`ImageEmbedding`. It embeds strings and `TextContent` as text and image `ResourceContent` as images
before storing. When searching, the helper turns the same query kinds back into vectors, applies
optional Maximal Marginal Relevance reranking with `mmr_vector_similarity_search` when
`rerank=True`, and always returns plain models - the low-level `QdrantResult` objects stay internal.
Calling `VectorIndex.search(...)` without a `query` falls back to `Qdrant.fetch(...)`.

The helper does not provision collections - `Qdrant.create_collection(...)` has to be called for
every indexed model before the first `VectorIndex.index(...)`, otherwise Qdrant rejects the write
with `Collection <Model> doesn't exist`.

```python
from draive import VectorIndex
from draive.qdrant import Qdrant, QdrantVectorIndex

async with ctx.scope(
    "qdrant-index",
    QdrantVectorIndex.prepare(),
    disposables=[QdrantClient()],
):
    # required once per model, the index does not create collections
    await Qdrant.create_collection(Document, vector_size=1536)
    await VectorIndex.index(
        Document,
        values=[Document(id="doc-3", text="hello")],
        attribute=Document._.text,
    )
    hits = await VectorIndex.search(Document, query="hello", limit=5, rerank=True)
```

Use `VectorIndex.delete(...)` to drop the stored embeddings for a given requirement set, reuse
`AttributeRequirement` logic for runtime filtering, and lean on the same `Qdrant` state for paging.

## Combining storage with workflows

Combine these helpers inside `ctx.scope(...)` to keep vector storage, configuration snapshots, and
retrievals aligned with the `haiway` lifecycle. The request-scoped state manages `AsyncQdrantClient`
connections, so you can spin up multiple contexts in the same process, reusing environment-variable
configuration as needed.
