# Postgres integrations

Draive ships with Postgres-backed implementations for common persistence interfaces so you can plug
relational storage into your workflows without writing adapters. All helpers live in
`draive.postgres` and reuse the shared `haiway.postgres.Postgres` connection states.

Current adapters include `PostgresConfigurationRepository`, `PostgresTemplatesRepository`,
`PostgresVectorIndex`, `PostgresConversationMemory`, and `PostgresAgentMemory`.

## Bootstrapping the Postgres context

Before using any adapter ensure a connection pool is available inside your context scope. The
helpers lean on `PostgresConnectionPool` and the `Postgres` facade exported from `draive.postgres`.

```python
from draive import ctx
from draive.postgres import (
    PostgresConnectionPool,
    PostgresConfigurationRepository,
    PostgresTemplatesRepository,
)

async with ctx.scope(
    "postgres-demo",
    PostgresConfigurationRepository.prepare(),  # use postgres configurations
    PostgresTemplatesRepository.prepare(),  # use postgres templates
    disposables=(
        PostgresConnectionPool.of(dsn="postgresql://draive:secret@localhost:5432/draive"),
    ),
):
    ...
```

Each adapter relies on the same connection scope, so you can freely mix them within a single
context.

The `postgres` extra (`pip install draive[postgres]`) pulls in `haiway[postgres]` and `asyncpg`,
which is everything the adapters need. pgvector-backed components require only the `vector`
**extension installed in the database** - no additional Python package. asyncpg ships no codec for
the `VECTOR` type, so `PostgresVectorIndex` binds every vector in pgvector's own text
representation and casts it in the statement, which needs no client-side registration.

If your deployment does need per-connection setup for unrelated reasons,
`PostgresConnectionPool.of(...)` accepts an `initialize` callback invoked for each new connection:

```python
from asyncpg.connection import Connection

from draive.postgres import PostgresConnectionPool


async def initialize_connection(connection: Connection) -> None:
    await connection.execute("SET search_path TO embeddings, public;")

PostgresConnectionPool.of(
    dsn="postgresql://draive:secret@localhost:5432/draive",
    initialize=initialize_connection,
)
```

## ConfigurationRepository implementation

`PostgresConfigurationRepository` persists configuration snapshots inside a `configurations` table
and keeps a bounded LRU cache to avoid repeated fetches. The table must expose the schema used in
the implementation:

```sql
CREATE TABLE configurations (
    identifier TEXT NOT NULL,
    name TEXT NOT NULL,
    content JSONB NOT NULL,
    created TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (identifier, created)
);

CREATE INDEX IF NOT EXISTS configurations_idx ON configurations (identifier, created DESC);
```

`await PostgresConfigurationRepository.migrate()` creates both within an active connection scope.

Key capabilities:

- `configurations(config)` returns every known identifier using cached results, narrowed to a single
    configuration type when one is given (default 10 minute TTL).
- `load(config, identifier=..., default=..., required=...)` fetches the newest JSON document per
    identifier and parses it into the requested configuration type.
- `define(config)` upserts a new configuration snapshot and clears both caches, guaranteeing fresh
    reads on the next call.
- `remove(identifier)` deletes all historical snapshots for the identifier and purges caches.

Tune memory pressure through `cache_limit` and `cache_expiration` arguments when instantiating the
repository.

Stored configurations are resolved explicitly - provider adapters read their configuration from the
scope state (`ctx.state(OpenAIResponsesConfig)`), never from a repository. Load a snapshot and place
it in the scope to apply it:

```python
from draive import ConfigurationRepository, ctx
from draive.openai import OpenAIResponsesConfig

config = await ConfigurationRepository.load(OpenAIResponsesConfig, required=True)

with ctx.updating(config):
    ...  # generation within this scope uses the loaded configuration
```

`Configuration.load(...)` resolves the repository entry first and falls back to `ctx.state(...)`
when the repository holds nothing for the identifier.

Snapshots are stored under the configuration class name by default. Pass an identifier as the first
argument to keep several variants of the same type side by side:

```python
await ConfigurationRepository.define(OpenAIResponsesConfig(model="gpt-5.5"))  # class name
await ConfigurationRepository.define("staging", OpenAIResponsesConfig(model="gpt-5.5"))
staging = await OpenAIResponsesConfig.load(identifier="staging", required=True)
```

## Custom migrations

Beyond the per-helper `migrate()` methods, `Postgres.execute_migrations(...)` runs application
migrations in order and records applied versions in a `migrations` table, so re-running is a no-op.
Each migration is a coroutine receiving the connection it runs on, within its own transaction:

```python
from draive import ctx
from draive.postgres import Postgres, PostgresConnection, PostgresConnectionPool


async def migration_0(connection: PostgresConnection) -> None:
    await connection.execute("CREATE TABLE IF NOT EXISTS documents (id UUID PRIMARY KEY)")


async def migration_1(connection: PostgresConnection) -> None:
    await connection.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS title TEXT")


async with ctx.scope(
    "migrations",
    disposables=(PostgresConnectionPool.of(dsn="postgresql://draive:secret@localhost:5432/draive"),),
):
    await Postgres.execute_migrations((migration_0, migration_1))
```

Migrations are identified by position, so they have to keep their order - a renamed or reordered
migration is reported as drift instead of being applied silently.

## TemplatesRepository implementation

`PostgresTemplatesRepository` mirrors the behaviour of the file-backed templates repository while
storing revisions inside a dedicated `templates` table:

See the [Templates](./Templates.md) guide for authoring patterns and runtime resolution examples.

```sql
CREATE TABLE templates (
    identifier TEXT NOT NULL,
    description TEXT DEFAULT NULL,
    content TEXT NOT NULL,
    variables JSONB NOT NULL DEFAULT '{}'::jsonb,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (identifier, created)
);

CREATE INDEX IF NOT EXISTS templates_idx ON templates (identifier, created DESC);
```

`await PostgresTemplatesRepository.migrate()` creates both within an active connection scope.

Capabilities:

- `templates(pagination)` returns a `Paginated[TemplateDeclaration]` reflecting the newest revision
    per identifier.
- `resolve(template)` and `resolve_str(template)` reuse a cached loader keyed by identifier to pull
    the latest template body before rendering arguments.
- `define(declaration, content=...)` persists a new revision, invalidates the loader cache, and
    ensures subsequent reads see the updated payload.

Use this adapter whenever your multimodal templates live alongside other structured content in
Postgres and you want on-demand caching with revision history.

## VectorIndex implementation (pgvector)

The `PostgresVectorIndex` helper persists dense embeddings in Postgres using the
[pgvector](https://github.com/pgvector/pgvector) extension. Each indexed `State` maps to its own
table named after the model class. The name is validated and interpolated unquoted, so Postgres
folds it to lower case (`Chunk` → `chunk`, `DocumentChunk` → `documentchunk`).

### Enable pgvector and create tables

Install the extension once per database and create a table for every data model you plan to index.
The implementation writes `embedding`, `payload`, `meta` and `created`, so those columns have to be
present. For a `Chunk` model the migration could look like this (adjust the vector dimension to
match your embedding provider):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunk (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    embedding VECTOR(1536) NOT NULL,
    payload JSONB NOT NULL,
    meta JSONB NOT NULL,
    created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Optional ANN index (requires pgvector >= 0.4.0)
CREATE INDEX IF NOT EXISTS chunk_embedding_idx
    ON chunk
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

The helper stores the serialized `State` instance inside `payload`, so the JSON schema mirrors
the model definition - including declared attribute aliases, which `AttributeRequirement` filters
resolve automatically (`Chunk._.identifier` filters the stored `id` key). It writes monotonically increasing `created` timestamps to preserve insertion
order for non-similarity queries.

### Wiring the index

Construct the index with `PostgresVectorIndex.prepare()` and reuse the shared `Postgres` state
inside an active context scope. The optional `mmr_multiplier` argument controls how many rows are fetched
before applying Maximal Marginal Relevance re-ranking when `rerank=True`.

```python
from collections.abc import Sequence
from typing import Annotated

from draive import Alias, State, VectorIndex, ctx
from draive.postgres import PostgresConnectionPool, PostgresVectorIndex


class Chunk(State):
    identifier: Annotated[str, Alias("id")]
    text: str


async with ctx.scope(
    "pgvector-demo",
    PostgresVectorIndex.prepare(),
    disposables=(
        PostgresConnectionPool.of(dsn="postgresql://draive:secret@localhost:5432/draive"),
    ),
):
    await VectorIndex.index(
        Chunk,
        values=[Chunk(identifier="doc-1", text="hello world")],
        attribute=Chunk._.text,
    )

    results: Sequence[Chunk] = await VectorIndex.search(
        Chunk,
        query="hello",
        limit=3,
        score_threshold=0.6,  # optional cosine similarity cutoff
        rerank=True,
    )
```

Queries can be strings, `TextContent`, `ResourceContent` (text or image), or pre-computed vectors.
When `score_threshold` is provided the helper converts it to the cosine distance cutoff used by
pgvector. Set `rerank=False` to return rows ordered solely by the database similarity operator.

### Payload filtering and requirements

Search and deletion accept `AttributeRequirement` instances which are evaluated against the stored
payload JSON. Requirements are translated to SQL expressions with both the attribute path and the
compared value bound as parameters, so `AttributeRequirement.equal` becomes
`payload #> $1::TEXT[] = $2::JSONB`. Using the `#>` accessor keeps the `jsonb` type, which is what
allows non-string values (integers, floats, booleans, UUIDs, timestamps) to be compared at all. The
`text_match` operator is not implemented yet and raises `NotImplementedError`, keeping the query
surface explicit.

## AgentMemory implementation

`PostgresAgentMemory` persists agent conversation context across turns, keyed by the executing
agent URI and the conversation thread identifier - both carried on the `AgentThread` passed to
each memory operation. See the [Agents](./Agents.md) guide for how `AgentMemory` participates in
agent execution.

Run the schema migration once with an acquired connection bound in context:

```python
from draive import ctx
from draive.postgres import Postgres, PostgresAgentMemory, PostgresConnectionPool

async with ctx.scope(
    "migration",
    disposables=(
        PostgresConnectionPool.of(dsn="postgresql://draive:secret@localhost:5432/draive"),
    ),
):
    async with Postgres.acquire_connection() as connection:
        with ctx.updating(connection):
            await PostgresAgentMemory.migrate()
```

Then create a memory instance and pass it to an agent:

```python
from draive import Agent
from draive.postgres import PostgresAgentMemory

assistant = Agent.generative(
    "assistant",
    instructions="You are a concise support assistant.",
    memory=PostgresAgentMemory.instance(),
)
```

Storage semantics:

- Context is stored as immutable snapshots: every remember inserts the full context as a new row
    and recall reads back the latest one whole, so steps may compact, summarize, or replace the
    context freely between turns.
- Persistence is lock-free and write-only: previous snapshots are never modified or deleted and
    remain available for tracking and verification. Snapshot history grows without bound over the
    lifetime of a thread.
- A single memory instance can serve multiple agents; recalled and remembered context is isolated
    per agent URI and thread, both taken from the executing `AgentThread` at call time.

## Putting it together

Combine these adapters with higher-level Draive components to centralise operational data in
Postgres. For example, wire the configuration repository into your configuration state, keep
reusable instruction sets shareable across teams, and persist model interactions for analytics—all
while letting `haiway` manage connection pooling and logging through the active context.
