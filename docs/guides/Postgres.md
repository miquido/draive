# Postgres integrations

Draive ships with Postgres-backed implementations for the most common persistence interfaces so you
can plug relational storage into your workflows without writing adapters. All helpers live in
`draive.postgres` and reuse the shared `haiway.postgres.Postgres` connection states.

## Bootstrapping the Postgres context

Before using any adapter ensure a connection pool is available inside your context scope. The
helpers lean on `PostgresConnectionPool` and the `Postgres` facade exported from `draive.postgres`.

```python
from draive import ctx
from draive.postgres import (
    Postgres,
    PostgresConnectionPool,
    PostgresConfigurationRepository,
    PostgresInstructionsRepository,
    PostgresModelMemory,
)

async with ctx.scope(
    "postgres-demo",
    PostgresConfigurationRepository(), # use use postgres configurations
    PostgresInstructionsRepository(), # use postgres instructions
    disposables=(
        PostgresConnectionPool.of(dsn="postgresql://draive:secret@localhost:5432/draive"),
    ),
):
    session_memory = PostgresModelMemory("demo-session")
```

Each adapter relies on the same connection scope, so you can freely mix them within a single
context.

## ConfigurationRepository implementation

`PostgresConfigurationRepository` persists configuration snapshots inside a `configurations` table
and keeps a bounded LRU cache to avoid repeated fetches. The table must expose the schema used in
the implementation:

```sql
CREATE TABLE configurations (
    identifier TEXT NOT NULL,
    content JSONB NOT NULL,
    created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (identifier, created)
);
```

Key capabilities:

- `configurations()` returns every known identifier using cached results (limit 1, default 10 minute
  TTL).
- `load(config, identifier)` fetches the newest JSON document per identifier and parses it into a
  requested configuration type.
- `load_raw(identifier)` fetches raw Mapping for given identifier.
- `define(config)` upserts a new configuration snapshot and clears both caches, guaranteeing fresh
  reads on the next call.
- `remove(identifier)` deletes all historical snapshots for the identifier and purges caches.

Tune memory pressure through `cache_limit` and `cache_expiration` arguments when instantiating the
repository.

## InstructionsRepository implementation

`PostgresInstructionsRepository` mirrors the behaviour of the in-memory instructions repository
while persisting values in a dedicated `instructions` table:

```sql
CREATE TABLE instructions (
    name TEXT NOT NULL,
    description TEXT DEFAULT NULL,
    content TEXT NOT NULL,
    arguments JSONB NOT NULL DEFAULT '[]'::jsonb,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (name, created)
);
```

Highlights:

- `available_instructions()` returns structured `InstructionsDeclaration` objects with cached
  results for quick catalog views.
- `resolve(instructions, arguments)` resolves the latest instruction body, leveraging a dedicated
  cache keyed by name and utilizing the provided arguments.
- `load(instructions)` loads the raw latest instructions keyed by name.
- `define(instructions, content)` stores new revisions and invalidates caches so subsequent reads
  return the fresh version.
- `remove(instructions)` removes all revisions for the instruction and drops relevant cache entries.

This adapter is ideal when you author system prompts and tool manifests centrally and want version
history per instruction.

## ModelMemory implementation

`PostgresModelMemory` enables durable conversational memory by persisting variables and context
elements in three tables sharing the same identifier:

```sql
CREATE TABLE memories (
    identifier TEXT NOT NULL,
    created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (identifier)
);

CREATE TABLE memories_variables (
    identifier TEXT NOT NULL REFERENCES memories (identifier) ON DELETE CASCADE,
    variables JSONB NOT NULL DEFAULT '{}'::jsonb,
    created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memories_elements (
    identifier TEXT NOT NULL REFERENCES memories (identifier) ON DELETE CASCADE,
    content JSONB NOT NULL,
    created TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

Capabilities:

- `recall(limit=...)` fetches the latest variables and replayable context elements (inputs/outputs)
  respecting the optional `recall_limit` supplied to the factory.
- `remember(*items, variables=...)` persists new context elements and optionally a fresh variable
  snapshot in a single transaction.
- `maintenance(variables=...)` ensures the base `memories` row exists and can seed default variables
  without appending messages.

Use the memory helper when you need stateful chat sessions, per-user progressive profiling, or
auditable interaction logs. Set `recall_limit` to bound the amount of context loaded back into
generation pipelines.

## Putting it together

Combine these adapters with higher-level Draive components to centralise operational data in
Postgres. For example, wire the configuration repository into your configuration state, keep
reusable instruction sets shareable across teams, and persist model interactions for analytics—all
while letting `haiway` manage connection pooling and logging through the active context.
