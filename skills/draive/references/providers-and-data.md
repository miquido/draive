# Providers, Claude Code, MCP, embeddings, datastores, AWS, evaluation

Install what you use: `pip install draive[<extra>]`; importing `draive.<pkg>` without its extra
raises an `ImportError` naming it. Extras: `openai` (+`openai_realtime`), `anthropic`
(+`anthropic_bedrock`), `mistral`, `gemini`, `ollama`, `systemone`, `vllm`, `cohere` (+`cohere_bedrock`),
`bedrock`, `aws`, `mcp`, `qdrant`, `surrealdb` (+`surrealdb_embedded`; imports as `draive.surreal`),
`postgres`, `opentelemetry`, `httpx`, `starlette`, `fastapi`, `rabbitmq`.

## Provider adapters

An adapter is a **disposable**: entering it installs capability states (`GenerativeModel`,
`TextEmbedding`, `DecisionModel`, ...) in the scope, so the rest of the code stays provider-agnostic
(`TextGeneration`, `ModelGeneration`, `DecisionGeneration`, `Conversation`, `TextEmbedding`). The model is a
`Configuration` **state in the scope**; adapters resolve `config or ctx.state(XConfig)` per call, so
a missing config is a wiring error and a per-call override is `generate(..., config=XConfig(...))`.

```python
from draive import TextGeneration, ctx, load_env
from draive.openai import OpenAI, OpenAIResponsesConfig

load_env()
async with ctx.scope("app", OpenAIResponsesConfig(model="gpt-5.5"), disposables=(OpenAI(),)):
    text = await TextGeneration.generate(instructions="Be brief", input="Hi")
```

| Adapter (wiring) | Env | Installs | Configs |
| --- | --- | --- | --- |
| `OpenAI()` | `OPENAI_API_KEY/BASE_URL/ORGANIZATION`; Azure `AZURE_OPENAI_API_KEY/API_BASE/API_VERSION/DEPLOYMENT_NAME` | `GenerativeModel`, `RealtimeGenerativeModel`, `TextEmbedding`, `ImageGeneration`, `GuardrailsModeration` | `OpenAIResponsesConfig`, `OpenAIEmbeddingConfig`, `OpenAIRealtimeConfig`, `OpenAIImageGenerationConfig`, `OpenAIModerationConfig` |
| `Anthropic()`, `Anthropic("bedrock", aws_region=)` | `ANTHROPIC_API_KEY`, `AWS_BEDROCK_REGION` | `GenerativeModel` | `AnthropicConfig` |
| `Mistral()` | `MISTRAL_API_KEY`, `MISTRAL_SERVER_URL` (or `MISTRAL_ENDPOINT`) | `GenerativeModel`, `TextEmbedding`, `GuardrailsModeration` | `MistralChatConfig`, `MistralEmbeddingConfig`, `MistralModerationConfig` |
| `Gemini()`, `Gemini(vertexai=True)` | `GEMINI_API_KEY` | `GenerativeModel`, `RealtimeGenerativeModel`, `TextEmbedding`, `DecisionModel` (not by default on Vertex AI, where the Decisions API is unavailable) | `GeminiConfig`, `GeminiEmbeddingConfig`, `GeminiDecisionConfig` |
| `Ollama()` | `OLLAMA_SERVER_URL` or `OLLAMA_HOST` | `GenerativeModel`, `TextEmbedding`, `DecisionModel` (needs a pulled decision model, text only) | `OllamaChatConfig`, `OllamaEmbeddingConfig`, `OllamaDecisionConfig` |
| `SystemOne(server_url=, images=)` | `SYSTEMONE_SERVER_URL` (default TypeSafe), `SYSTEMONE_API_KEY` or `TYPESAFE_API_KEY` | `DecisionModel` | `SystemOneConfig` |
| `VLLM(base_url=)` | `VLLM_BASE_URL` (else `OPENAI_BASE_URL`, one is required), `VLLM_API_KEY` | `GenerativeModel`, `TextEmbedding` | `VLLMChatConfig`, `VLLMEmbeddingConfig` |
| `Bedrock(aws_region=)` | `AWS_BEDROCK_REGION` | `GenerativeModel`, `GuardrailsModeration` | `BedrockChatConfig`, `BedrockInputGuardraisConfig`, `BedrockOutputGuardraisConfig` |
| `Cohere()`, `Cohere("bedrock", aws_region=)` | `COHERE_API_KEY`, `COHERE_SERVER_URL` | `TextEmbedding`, `ImageEmbedding` (no chat) | `CohereTextEmbeddingConfig`, `CohereImageEmbeddingConfig` |

- `SystemOne` speaks the System One decision API (`/v1/systemone`) of TypeSafe Jev, Ollama,
  gateways (OpenRouter, ...) and OpenJev; pick image delivery per server with `images=`
  (`"assets"` TypeSafe default, `"images"` Ollama, `"inline"` OpenJev).
- `features=(...)` narrows what an adapter installs (`Mistral(features=(TextEmbedding,))`): chat
  from one provider, embeddings from another. Without it a later adapter's states replace the
  earlier one's of the same type.
- Configs are `State`s (`cfg.updating(max_output_tokens=3)`); unset options are `Missing`.
  `model` is required except for `OpenAIEmbeddingConfig`, `OpenAIRealtimeConfig`,
  `OpenAIImageGenerationConfig`, `OpenAIModerationConfig`, `MistralModerationConfig` and the
  decision configs (`GeminiDecisionConfig`, `OllamaDecisionConfig`, `SystemOneConfig`), which
  default it (`OpenAIRealtimeConfig` still requires `input_parameters`/`output_parameters`).
  `CohereTextEmbeddingConfig.purpose` is required (`"search_query"` / `"search_document"` / ...).
- Defaults that bite: `AnthropicConfig.max_output_tokens=16384` and thinking consumes it (a small
  limit ends a turn without an answer); `GeminiConfig.temperature=1.0` and every Gemini safety
  category is **off** unless `safety=GeminiSafetyConfig(...)`; Mistral reasoning models need
  `prompt_mode="reasoning"`; Ollama ones `thinking=...`; Bedrock takes an inline
  `guardrail_identifier`/`guardrail_version` on `BedrockChatConfig`.

### OpenAI strict structured output

A `State` output goes out as a **strict** `json_schema` (`openai/schema.py`): every property
required, `additionalProperties: false`, unsupported `format`/`uniqueItems`/`contentEncoding`
dropped. Tuples, recursive types and objects nested deeper than 10 fall back to an unenforced
schema (debug log "OpenAI strict output is unavailable"). So: a field that may be absent must be
`X | None` (strict makes the model fill every field); use `Sequence[T]` not tuples; avoid
recursive models and `Mapping[str, Any]`/`Meta` fields in outputs. Tool parameters are strict only
with `strict_parameters` set to true in the tool's `meta`; a free-form mapping argument leaves
that schema unenforced.

## Claude Code agent - `draive.claude`

Needs no extra, only the `claude` executable (the `mcp` extra for `tools=`). It runs Claude Code as a
fully managed process and exposes it as an `Agent`.

```python
from draive.claude import (
    ClaudeCode, ClaudeCodeApiKey, ClaudeCodeConfig, ClaudeCodeSessions, ClaudeCodeTelemetry, claude_code_agent,
)

async with ctx.scope(
    "app",
    ClaudeCodeConfig(model="sonnet", tools=("Read", "Grep", "Edit"), max_turns=20, directory="/project",
                     telemetry=ClaudeCodeTelemetry.from_env(signals={"metrics", "logs", "traces"})),
    disposables=(ClaudeCode(credential=ClaudeCodeApiKey("sk-ant-api...")),),   # or ClaudeCodeOAuthToken, env
):
    coder = claude_code_agent("coder", instructions="Fix bugs.",
                              sessions=ClaudeCodeSessions(state_directory="/var/lib/agent"))
    async for chunk in coder.call(input="Fix the failing test"): ...   # text + ProcessingEvents
```

`claude_code_agent(agent, *, description, instructions, sessions: ClaudeCodeSessions | None = None,
config, tools: Toolbox | Iterable[Tool] = Toolbox.empty, meta)`.

- Invalid states do not exist: `ClaudeCodeConfig` refuses non-positive limits, relative directories and
  reserved environment variables when created (`ValidationError`); a credential is one of
  `ClaudeCodeApiKey`/`ClaudeCodeOAuthToken`/`ClaudeCodeLocalLogin` (keys and tokens 1-4096 bytes);
  `compaction` is `ClaudeCodeAutoCompaction(percent=, window=)` or `ClaudeCodeNoCompaction()`;
  telemetry `signals` are a non-empty set.
- Authentication is only the given credential (handed over through a file descriptor, kept out of
  arguments, environment and reprs). Nothing of the user's login is reachable: private home, an
  environment built from scratch, and the run is stopped if Claude Code reports another
  authentication source. No credential means entering the scope fails. `ClaudeCodeLocalLogin` hands
  over the access token of a file based `claude /login` login (refreshed through Claude Code),
  `ClaudeCodeConfig(connectors=ClaudeCodeAccountConnectors())` turns on claude.ai connectors
  (`mcp__claude_ai_<Name>__*`, need that login and `allowed_tools`). Connectors are filtered by
  `ClaudeCodeAccountConnectors(excluded=("Slack",), excluded_urls=..., allowed=ClaudeCodeConnectorsAt(urls=...))`
  - an allowlist keeps `mcp_servers` (and the `draive` tools server) by name, so those names need
  letters, digits, `_` and `-` (else `ClaudeCodeUnavailable`) - and failed connectors are warned
  about, the ones needing authorization only recorded (`claude_code.connectors_unavailable`).
- The executable is resolved from `binary=` of `ClaudeCode`, `CLAUDE_CODE_BINARY` and `PATH` only.
- Isolated and deny-by-default: no built-in tools unless `tools=` of the config (file tools you add
  are scoped to the working and additional directories - no symlink or `..` escape - `Bash` is not
  confined, disposable environments only), `permission_mode="dontAsk"`, no setting sources (nor the
  working directory `CLAUDE.md`), no plugins/skills/auto memory, no MCP except `mcp_servers`. `Bash`
  and `bypassPermissions` only in a disposable environment.
- `ClaudeCodeConfig(claude_md=ClaudeCodeProjectClaudeMd(), directory=...)` loads only the
  `CLAUDE.md` of the working directory (no parents, no settings); the ones of `additional_directories`
  are excluded unless `additional_directories=ClaudeCodeLoadAdditionalClaudeMd()`.
- Sandbox: `ClaudeCodeConfig.sandbox` is `ClaudeCodeSandbox()` by default - with `Bash` in `tools` its
  commands run in Claude Code's sandbox (writes only in the working/additional directories, strict, Linux
  needs bubblewrap + socat + user namespaces; commands are still allowed by `allowed_tools` Bash rules, network blocked unless
  `allowed_domains`; tune with `ClaudeCodeSandbox(allowed_writes, denied_writes, denied_reads, allowed_reads,
  excluded_commands, allowed_domains, denied_domains)`, absolute paths); `ClaudeCodeNoSandbox()` opts out within a container, extra
  keys in `settings["sandbox"]`. Unsandboxed `Bash` can call draive tools directly (`disallowed_tools`
  not enforced against it).
- Isolation details: configured `enabledPlugins` and `claudeMdExcludes`, `allowedMcpServers` and `deniedMcpServers` in `settings` extend the
  derived ones (lists). `setting_sources=("project",)` loads `CLAUDE.md`/rules of the parents too, except
  the user's (`~/CLAUDE.md`, `~/CLAUDE.local.md`, `~/.claude`); outside a git repo it loads parent
  `.claude/agents` as well, the user's too under home (logged as a warning). Unknown built-in plugins
  are warned about, organization-enforced ones logged as info.
- draive tools: `tools=` of `claude_code_agent` serves a draive `Toolbox` to the run as the MCP server
  `draive` (tools `mcp__draive__<name>`, allowed automatically, run in the agent process with its
  `ctx` state) through a per-run private Unix socket and a stdlib relay started by Claude Code. Needs
  the `mcp` extra, only `response` handling, names up to 115 characters of `[A-Za-z0-9_-]` - refused
  at agent creation otherwise (`ValueError`). `draive` is a reserved `mcp_servers` key. Distinct from
  `ClaudeCodeConfig.tools` (built-in tools).
- Memory: only Claude Code sessions - `sessions=ClaudeCodeSessions(state_directory=)` resumes the
  session of each thread with the full context, also of an interrupted run; without it every call is
  fresh. `AgentMemory` is not supported. The state directory must be stable and persistent (a shared
  volume with replicas). Context injection and usage reading are not possible with Claude Code.
- Output: text blocks of the main agent are streamed as paragraphs (deltas are not repeated by their
  complete block) + `ProcessingEvent`s (`claude_code.tool_use`, `.tool_result`, `.subagent_text`,
  `.result`). Failures are variants of `ClaudeCodeFailed`: `ClaudeCodeErrored` (`failure.reason`,
  `api_error_status`, `api_error_type`, `terminal_reason`; `ClaudeCodeAuthenticationFailed` when the API
  rejected the credential, i.e. a revoked login token), `ClaudeCodeTimedOut`, `ClaudeCodeEnded` (`exit_code`,
  redacted `stderr`), `ClaudeCodeUnfinished` (output ended without a result while no process exit is
  known; `stderr` is `None` for a custom runner), `ClaudeCodeOutputTooLarge`; a missing
  binary/credential is `ClaudeCodeUnavailable` (`ClaudeCodeLoginUnusable` when a `ClaudeCodeLocalLogin`
  gives no valid token - log in again). Limits: `max_turns`,
  `max_budget_usd`, `timeout` (terminated gracefully so Claude Code stops its own commands, killed
  after `stop_grace`; on Linux detached commands which keep the run marker of their environment are
  stopped too, elsewhere they can outlive the run), `stop_grace`.
- Telemetry: `ClaudeCodeTelemetry` exports Claude Code's OTLP metrics/events (traces are beta) and
  continues the trace of the calling agent (`TRACEPARENT`); draive also records `model.*` usage
  (provider `claude_code`) and `agent.claude_code.cost`.
- Fake it with `ClaudeCodeRunner(binary="/fake", running=fn)` where `fn(launch)` yields
  `ClaudeCodeText`/`ClaudeCodeToolUse`/... and a final `ClaudeCodeResult(outcome=ClaudeCodeSuccess(text=))`.

## MCP - `draive.mcp`

```python
from draive import ToolsProvider, ctx
from draive.mcp import MCPClient, MCPClients, MCPServer

async with ctx.scope("app", disposables=(MCPClients(      # one disposable aggregating many servers
    MCPClient.stdio(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]),
    MCPClient.streamable_http(url="https://example.com/mcp", headers={"Authorization": "Bearer ..."}),
),)):
    toolbox = await ToolsProvider.toolbox(suggesting=True)    # all MCP tools; pass as tools=toolbox
```

- `MCPClient.stdio(identifier=None, *, command, args, env, features, tags)`, `.streamable_http(url=,
  headers=, timeout=30, ...)` (preferred remote transport), `.sse(url=, ...)` (deprecated). Entering
  installs `ToolsProvider` and `ResourcesRepository`; one client is a disposable on its own.
- Server tool schemas are translated into `ModelToolParametersSpecification` (`$defs` inlined,
  `allOf` combined, `oneOf`/root alternatives merged, value constraints such as `minimum` dropped -
  the server still enforces them). A tool whose schema cannot be translated faithfully is skipped
  with a warning, the rest of the server's tools still load.
- Serve your own: `MCPServer(name=, version=, instructions=, resources=, tools=Toolbox | [Tool],
  disposables=())`; `await server.run_stdio()`, or mount `server.prepare_streamable_http_asgi(
  path="/mcp", stateless=False)` (ASGI app; `prepare_asgi()` is the deprecated SSE variant). Its
  `disposables` are entered in the server lifespan, so tools reach their `State` through `ctx`.

## Embeddings, vector index, similarity, splitting

```python
from draive import State, TextEmbedding, VectorIndex, ctx, split_text
from draive.helpers import VolatileVectorIndex      # in-memory index: tests, small corpora
from draive.openai import OpenAI, OpenAIEmbeddingConfig

class Chunk(State):
    text: str

async with ctx.scope("rag", OpenAIEmbeddingConfig(), VolatileVectorIndex(), disposables=(OpenAI(),)):
    parts = split_text(document, part_size=800, count_size=len, separators="\n\n", part_overlap_size=80)
    await VectorIndex.index(Chunk, attribute=Chunk._.text, values=[Chunk(text=p) for p in parts])
    found = await VectorIndex.search(Chunk, query="refund", limit=4, score_threshold=0.6, rerank=True)
```

- `VolatileVectorIndex()` is a function returning a `VectorIndex` state. Indexes embed through the
  scope's `TextEmbedding`/`ImageEmbedding`, so an embedding adapter is required.
- `VectorIndex.index(Model, *, attribute=, values=)`: `attribute` is `Model._.path` or a callable
  yielding `str`, `TextContent` or an image `ResourceContent` (never mixed in one call).
  `search(Model, *, query=, score_threshold=, requirements=, limit=)` where `query` is `str |
  TextContent | ResourceContent | Sequence[float] | None` (none: list by `requirements`);
  `delete(Model, *, requirements=)`. `rerank=True` (MMR) is a backend `**extra`.
- Direct: `await TextEmbedding.embed("x")` -> `Embedded[str]` (`.value`, `.vector`);
  `embed_many([...])`; `embed(obj, attribute=Obj._.text)`; `ImageEmbedding.embed(bytes)`.
- Pure numpy helpers (sync, no scope) return **indices**: `vector_similarity_search(query_vector,
  values_vectors, limit=None, score_threshold=None)`, `mmr_vector_similarity_search(query_vector,
  values_vectors, limit=None, lambda_multiplier=0.5)`; `vector_similarity_score(a, b) -> float`.
  `cosine_similarity` lives in `draive.embedding`, not the root.
- `split_text(text, part_size, count_size, separators=None, part_overlap_size=None, mode="basic")`;
  `count_size` measures a part (`len` or a tokenizer count). Modes `"basic"`, `"exhaustive"`,
  `"exhaustive_regex"` (the only one accepting compiled patterns as separators).

## Datastores

Each is a scope disposable plus `.prepare()` states (`VectorIndex`, `TemplatesRepository`,
`ConversationMemory`, `ConfigurationRepository`); indexing still needs an embedding adapter.
Filters are `AttributeRequirement`s on typed paths (`Chunk._.text`, never strings).

**Postgres** (docs guide *Postgres*; pool rules in `integrations.md`)

```python
from draive.postgres import (PostgresConfigurationRepository, PostgresConnectionPool,
    PostgresTemplatesRepository, PostgresVectorIndex)

async with ctx.scope(
    "app",
    PostgresVectorIndex.prepare(mmr_multiplier=8),
    PostgresTemplatesRepository.prepare(),
    PostgresConfigurationRepository.prepare(),
    disposables=(PostgresConnectionPool.of(dsn="postgresql://u:p@host:5432/db", connection_limit=8),),
): ...
```

- Also `PostgresConversationMemory.prepare(thread=...)` and `PostgresAgentMemory.instance()` (for
  `Agent.generative(..., memory=...)`). Run each `X.migrate()` once with a pinned connection:
  `async with Postgres.acquire_connection() as c: with ctx.updating(c): await X.migrate()`.
- The vector index creates **nothing**: install the `vector` extension and a table per model named
  after the class lower-cased (`id, embedding VECTOR(n), payload JSONB, meta JSONB, created`; SQL in
  the guide). `n` must match the embedding model. Memory snapshots are append-only.

**Qdrant** (docs guide *Qdrant*)

```python
from draive.qdrant import Qdrant, QdrantClient, QdrantVectorIndex

async with ctx.scope("app", QdrantVectorIndex.prepare(), disposables=(QdrantClient(),)):
    await Qdrant.create_collection(Chunk, vector_size=1536)     # once per model, the index won't
    await VectorIndex.index(Chunk, attribute=Chunk._.text, values=chunks)
```

`QdrantClient(host, port, grpc_port, api_key, ssl, timeout)` falls back to `QDRANT_HOST/PORT/
GRPC_PORT/API_KEY`; gRPC is the default transport, so both ports must be reachable;
`QdrantClient(in_memory=True)` for tests. Low level: `Qdrant.create_collection(Model, *,
vector_size, distance="Cosine", skip_existing=True)`, `create_index`, `store(Model, objects=[
Embedded])`, `fetch(...)` -> `Paginated`, `search(Model, query_vector=, limit=8)`, `delete`;
failures raise `QdrantException`.

**SurrealDB** (3.x server, or `surrealdb_embedded` for `mem://`)

```python
from draive.surreal import SurrealClient, SurrealVectorIndex

async with ctx.scope(
    "db",
    SurrealVectorIndex.prepare(search_effort=40),
    disposables=(SurrealClient(url="ws://localhost:8000/rpc", namespace="app", database="app",
                               user="root", password="root"),),
):
    await SurrealVectorIndex.migrate(Chunk, dimensions=1536)    # table + HNSW index
```

Namespace and database must already exist, and a never-defined table cannot even be read, so run
every `migrate()` first (`SurrealVectorIndex.migrate(Model, *, dimensions, vector_type="F64",
distance="COSINE")`). Also `SurrealTemplatesRepository` and `SurrealConversationMemory`
(`prepare()` + `migrate()` each); failures raise `SurrealException`.

**AWS** (`aws` extra, `draive.aws`)

```python
from draive import Resource, ResourcesRepository, ctx
from draive.aws import AWS, AWSSQS, AWSCloudwatch

async with ctx.scope("app", disposables=(AWS(region_name="eu-west-1"),)):   # ambient credentials
    report = await ResourcesRepository.fetch("s3://bucket/reports/q1.json")
    await ResourcesRepository.upload(
        Resource.of(b"{}", uri="s3://bucket/out.json", mime_type="application/json")
    )
    async with await AWSSQS.queue("orders", content_encoder=encode, content_decoder=decode) as queue:
        await queue.publish(order)                    # MQQueue, used like the RabbitMQ one
    await AWSCloudwatch.put_metric(namespace="app", metric="orders", value=1)
```

- `AWS(region_name=, access_key_id=, secret_access_key=, profile_name=, features=)` installs
  `ResourcesRepository` (S3, `s3://bucket/key` URIs only, no deleting), `AWSSQS` and
  `AWSCloudwatch` (`put_log`, `put_metric`, `put_event`); narrow with `features=` - it replaces
  the default HTTP `ResourcesRepository` of the scope.
- SQS encoders/decoders work on `str` (RabbitMQ ones on `bytes`). `reject()` redelivers
  immediately: configure a dead-letter queue with `maxReceiveCount`, or a failing message loops.
- `CloudwatchObservability(log_level=, log_group=, log_stream=, event_bus=, event_source=,
  metrics_namespace=)` is an `observability=` for `ctx.scope`. Errors are `AWSError` >
  `AWSAccessDenied`, `AWSInvalidArgument`, `AWSResourceNotFound`.

## Evaluation - `draive.evaluation`, `draive.evaluators`

Evaluator = async callable returning an `EvaluationScore`; scenario = several evaluators on one
value; suite = a scenario over stored cases. Results: `score: float`, `threshold`, `passed`,
`performance` (% of threshold), `meta` (judge text under `"comment"`).

```python
from collections.abc import Sequence

from draive import Multimodal, MultimodalContent, State, TextGeneration
from draive.evaluation import (EvaluationScore, EvaluatorResult, EvaluatorSuiteCase, evaluate,
    evaluator, evaluator_scenario, evaluator_suite)
from draive.evaluators import groundedness_evaluator, readability_evaluator

@evaluator(name="keyword_presence", threshold=0.8)           # snake_case name; default threshold 1
async def keywords(content: Multimodal, /, *, required: Sequence[str]) -> EvaluationScore:
    text = MultimodalContent.of(content).to_str().lower()
    return EvaluationScore.of(sum(k in text for k in required) / len(required), meta={"comment": "..."})

@evaluator_scenario(name="answer_quality")
async def quality(answer: str, reference: str) -> Sequence[EvaluatorResult]:
    return await evaluate(answer,
        keywords.prepared(required=["refund"]),
        groundedness_evaluator.with_threshold("good").prepared(reference=reference),
        concurrent_tasks=2)

class Case(State, serializable=True):
    question: str

@evaluator_suite(Case, name="qa", storage=[EvaluatorSuiteCase(parameters=Case(question="..."))])
async def qa(parameters: Case) -> Sequence[EvaluatorResult]:
    answer = await TextGeneration.generate(input=parameters.question)
    return await evaluate(answer, readability_evaluator.prepared())

result = await qa()              # every case; qa(5) random, qa(0.3) a fraction, qa([identifiers])
print(result.passed, result.report(detailed=True, include_passed=False))
```

- The evaluated value is the first positional-only argument, the rest keyword-only. A direct call
  returns `EvaluatorResult`; a scenario `EvaluatorScenarioResult` (`.results`, `.passed`,
  `.report(detailed=)`); a suite `EvaluatorSuiteResult` (`.results` per case).
- Levels: `"none"` 0.0, `"poor"` .2, `"fair"` .4, `"good"` .6, `"excellent"` .8, `"perfect"` 1.0
  (floats/bools work too). Chain `.with_threshold`, `.with_meta`, `.with_name`, `.with_state`,
  `.contra_map(fn)`; combine with `Evaluator.highest/lowest/average`; `.referenced(reference=...)`
  scores an evaluator against human labels (see also `cohen_kappa`, `reference_conformance`).
- Catalog: `<name>_evaluator` plus a `<name>_context_evaluator` twin (takes a `ModelContext`) for
  coherence, completeness, conciseness, consistency, coverage, creativity, expectations,
  factual_accuracy, fluency, groundedness, helpfulness, jailbreak, readability, relevance, safety,
  similarity, tone_style, truthfulness, required_keywords, forbidden_keywords; without twin:
  `text_/image_vector_similarity_evaluator`, `cohen_kappa_evaluator`; context only:
  `tool_usage_context_evaluator`. Required arguments are tabled in the docs guide
  *EvaluatorCatalog*.
- Catalog evaluators are LLM judges calling the scope's `GenerativeModel`; code-only ones need none.
- Suite storage: in-memory cases, a `Path`/`str` JSON file (running against a missing file raises;
  `add_case` creates it) or an `EvaluatorSuiteCasesStorage`. `add_case`, `cases()`,
  `generate_cases(count=, examples=, guidelines=, persist=)` (a model call per case),
  `with_storage`, `concurrent_evaluations=`.

## Haiway integrations through `draive.*`

`draive.httpx`, `draive.opentelemetry`, `draive.starlette`, `draive.fastapi`, `draive.postgres`
re-export the haiway objects described in `integrations.md` (same classes):

```python
from draive.fastapi import ServerContext, application
from draive.opentelemetry import OpenTelemetry

context = ServerContext(OpenAIResponsesConfig(model="gpt-5.5"), disposables=(OpenAI(),),
                        observability=OpenTelemetry.observability)       # callable, uncalled
app = application(context, routers=(router,), exception_handler=handle_500)
```

- `application(context, /, *, routers=, middleware=, exception_handler=, **extra)` takes the
  unhandled-error handler as `exception_handler` (`(Request, Exception)` callable returning a
  `Response`; there is no `ExceptionHandling` type). The Starlette variant takes `routes=`. Also
  exported: `StreamResponse`, `StateResponse`, `ContextMiddleware`, `request_trace_context`,
  `ObservabilityPreparing`.
- Adapters in the context's `disposables` are entered once at startup, inherited by every request.
  `HTTPXClient(base_url=, headers=, timeout=5)` from `draive.httpx`;
  `OpenTelemetry.configure(service=, version=, environment=, otlp_endpoint=)` once per process.
