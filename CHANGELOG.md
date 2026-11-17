# Changelog

## 0.120.0

### Changed

- Requires Python 3.15 and haiway 0.62.1.
- `draive.utils.gather_all` is removed in favor of haiway `gather` (exported from `draive`).
  Several coroutines failing together now raise an `ExceptionGroup` instead of the first failure.
- `draive.fastapi` no longer exports `ExceptionHandling` (removed in haiway 0.62.1), pass the
  unhandled error handler to `application(..., exception_handler=...)` as a plain callable.
- Resource templates use haiway `URITemplate` for expansion, matching and verification.
  `ResourceReferenceTemplate.template_uri` is a `URITemplate` (serialized as
  `{"template": "..."}`), `@resource(uri_template=...)` accepts a string or a `URITemplate`.
  URIs carrying undeclared query parameters (for templates without a query) or encoded
  separators within host variables are refused, a missing required variable fails the
  expansion, aliased variables expand correctly.
- Resource template variables always have to match declared function arguments, a variadic
  `**kwargs` no longer makes the declaration accept an undeclared variable. It remains available
  for extra arguments of `resolve(...)`.
- Public re-exports from haiway now include `URITemplate`, `DefaultEnv`, `DefaultFactory`,
  `Directory`, `PaginationToken`, `Paths`, `ScopeKind`, `gather` and `parse_bool`.
- Mistral reads `MISTRAL_ENDPOINT` as an alias for `MISTRAL_SERVER_URL`.
- SurrealDB 3.x or newer is required (verified with 3.2 and 4.0).

### Added

- `ToolArgumentsException` is exported from `draive` and `draive.tools`.
- `skills/draive` - an Agent Skill for coding agents writing Draive code.
- Telemetry: provider failure/rate-limit metrics, embedding token usage, tool invocations,
  guardrail verdicts, evaluator suite case errors and durations, Qdrant and SurrealDB
  operation metrics, per-root AWS CloudWatch traces.
- `ConversationException`; iteration limits raising typed exceptions: `max_iterations` for
  agents and steps, a fixed limit of 32 for conversation completion.
- `docs/guides/Resources.md`.
- `GenerationOutputMissing` raised by image and audio generation instead of a bare
  `ValueError`; `AWSInvalidArgument`.

### Fixed

- MCP resource URIs survive the client identifier round trip: custom schemes without an
  authority (`notes:///docs/a`) keep their `//`, and dots within client identifiers are escaped
  (`my%2Eserver`) so listed URIs route back to the client which listed them unambiguously.
- Ollama tool calls no longer emit a Pydantic serializer warning for open tool parameters.
- Provider adapters translate SDK failures into typed model exceptions consistently
  (OpenAI, Anthropic, Mistral, Gemini, Ollama, Cohere, vLLM, Bedrock).
- Evaluation: invalid score bounds and empty suites are rejected, suite case errors become
  failed results, evaluator signature errors surface. Out-of-range evaluator scores become
  errored zero results, `EvaluationScore` ordering accepts ints, `Evaluator.average` is
  rounding-safe, suite file storage bootstraps a missing file on `add_case` (written
  atomically, a missing file raises `FileNotFoundError` on load), `add_case` rejects duplicate
  identifiers with `ValueError`, scenario names with spaces are normalized, brief case
  reports format performance.
- Guardrails: `GuardrailsFailure` from a moderation checker stays a failure. Bedrock guardrails
  report topic, PII, word and contextual grounding violations. Vector similarity evaluators
  absorb float error for identical vectors and score NaN as zero.
- Postgres, Qdrant and SurrealDB integrations: transactions, batching, MMR overfetch,
  quoting and error types.
- Unsupported multimodal elements and non-multimodal tool results raise typed errors.
- Toolbox: a tool cancelled from within still answers its request; `filtered(tools="name")`
  matches exact names. Skills load from paths below symlinked directories. Rate limit delays
  are finite and non-negative. `ModelSessionEvent` turn payloads round-trip JSON. Realtime
  conversations keep memory order across tool calls and deliver tool events while the
  provider is silent; templated conversation messages keep multimodal arguments.
- Tools: argument validation failures (invalid, missing or unexpected argument) reach the model
  with an actionable message, other tool errors stay generic. `Toolbox.handle` returns
  responses in request order while still executing concurrently and streaming events.
  Input validation in `Toolbox`, `AgentMemory.volatile` and `ModelGeneration.generate` raises
  `ValueError`/`TypeError` instead of asserting. `model_rate_limit` honours HTTP-date
  `Retry-After`. Skill loading tolerates a leading BOM and ignores unknown frontmatter keys.
- Realtime conversations: a failed or cancelled tool handler answers its request with an error
  response, an assistant turn interrupted by a user turn is remembered separately, empty turns
  are not remembered. OpenAI realtime completes each assistant item once, sends one
  `response.create` per tool-call batch, tolerates benign error events and malformed
  tool arguments or audio (typed `ModelOutputInvalid`). Gemini live drops interrupted partial
  output and keeps generated function-call identifiers stable.
- Providers: OpenAI Responses maps in-stream rate limits to `ModelRateLimit`, fails on a
  missing terminal event, separates reasoning summary parts. Gemini 503/UNAVAILABLE and
  Bedrock stream `serviceUnavailable` are retryable. Anthropic and Bedrock skip blank text
  blocks and empty turns, Bedrock keeps role alternation. Mistral and vLLM put tool messages
  before user content, Mistral groups parallel tool calls in one assistant message. Foreign
  reasoning blocks are dropped on replay instead of raising (OpenAI, Anthropic, Gemini).
- Evaluation: scenarios raise on invalid arguments, `EvaluatorResult.performance` is 100% for
  zero thresholds, `Evaluator.average` keeps errors of failed inputs, results compare
  with foreign types via `NotImplemented`, keyword matching uses `casefold`, reports escape XML.
  Bedrock guardrails reject unsupported media with `GuardrailsFailure` and skip empty content.
  `suite(n)` int is a count, float a fraction, bool is rejected.
- Templates and steps: file-backed `TemplatesRepository` serializes access and writes
  atomically; `prepare_instructions` teaches `{%variable%}` placeholders; splitters respect
  `part_size` with overlap; `with_timeout` only reports its own timeout and rejects
  non-finite or non-positive limits; resource template parameters keep text for string unions;
  embedding search/MMR handle negative limits, NaN and empty vectors; `ArtifactContent`
  `get_int` rejects bool, `get_float` accepts int, `to_str` keeps unicode readable,
  `of(hidden=False)` un-hides; `cosine_similarity` never leaks NaN for infinite input.
- MCP server: readable validation message for missing tool arguments, `text/*` and `+json`
  resources are served as text, resource templates of mixed arities type-check.
- AWS: S3 references percent-encode keys, failures are translated consistently, failed downloads
  leave no partial file and written files get the umask-default mode, `list(limit=0)`
  short-circuits, non-ASCII metadata raises `AWSInvalidArgument`. SQS translates missing queues,
  formats booleans, accepts `http(s)` queue URLs, `reject()` releases the message immediately
  (configure a dead-letter queue). Observability attribute sanitizing never raises.
- Qdrant: `limit=0` returns nothing, `None` in equal/not-equal requirements is supported.
- Realtime conversation memory remembers tool requests and responses (as `ConversationEvent`
  entries), so recalled context matches what the model saw.
- Conversation memory recall merges adjacent same-kind context elements (inputs or outputs)
  into one, so providers always get a valid input/output alternation after interrupted or
  empty turns.
- OpenAI realtime requests the follow-up response only after the originating response finished
  and all of its tool calls were answered. Realtime conversations wait for tool output only
  while a tool is running. AWS observability reports unsupported attribute values by type name
  only. `Evaluator.average` ignores non-text `error` meta. `EvaluatorSuite.add_case`
  bootstraps only the built-in file storage, a custom storage raising `FileNotFoundError`
  propagates. The MCP server serves undecodable text resources as blobs.
- Gemini Live coalesces adjacent audio chunks of a completed turn into a single resource block
  (the live stream still emits every chunk). Bedrock raises `ValueError` when the closing model
  input has no deliverable content instead of sending a request ending on an assistant message.
- vLLM materializes message content so SDK request retries no longer send empty tool or user
  content, sends empty tool results as an empty string and raises `ModelOutputInvalid` for
  non-object tool arguments. Ollama sends tool messages before user content and skips empty
  assistant messages. vLLM and Ollama failure reasons are never empty. Anthropic tool-result
  content is materialized as well (retry-serialization pin tests for Anthropic and OpenAI).
- Mistral: non-object tool arguments raise `ModelOutputInvalid`, moderation keeps the highest
  score per category across inputs. Cohere on Bedrock: throttling and capacity errors raise
  `ModelRateLimit`, validation errors `ModelInputInvalid`. Mistral and Cohere embeddings raise
  `ModelOutputInvalid` for inconsistent responses instead of `ValueError`.
  Mistral HTTP errors report only the status and the short provider message, never the
  response body.
- Realtime conversation memory remembers tool exchanges observed from the session (OpenAI
  Realtime) and flushes pending assistant output on session close and once a tool-using turn
  completes (Gemini Live), deduplicating requests and responses by identifier. User text
  written through `write()` is still not remembered.

### Breaking behavior notes

- S3 `s3://bucket/key` references are percent-decoded when parsed. References stored by
  earlier releases whose keys contain a literal `%XX` sequence must be re-encoded.
- Tool responses of `Toolbox.handle` follow request order, evaluator scenarios raise on
  invalid arguments, S3 upload raises `AWSInvalidArgument` for non-ASCII metadata, Bedrock
  guardrail violation keys carry a category prefix (`topic:`, `pii:`, `word:`, `grounding:`),
  `ModelSessionEvent.turn_completed` accepts only `ModelInput | ModelOutput`.
