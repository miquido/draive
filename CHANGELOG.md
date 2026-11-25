# Changelog

## 0.123.0

### Added

- `draive.claude` can use the connectors of a claude.ai account (cloud MCP servers, tools named
  `mcp__claude_ai_<Name>__*`) per run: `ClaudeCodeConfig.connectors` takes
  `ClaudeCodeAccountConnectors(wait=True)`, the default is `ClaudeCodeNoConnectors()`. Connectors
  need a credential with the `user:mcp_servers` scope, a run with any other one fails with
  `ClaudeCodeUnavailable` instead of silently having no connector tools.
- `ClaudeCodeLocalLogin`, a credential of a local Claude Code login (`claude /login`) read from
  `<config_directory>/.credentials.json` (logins kept in the keychain (macOS) are not supported). A
  token about to expire is refreshed under a lock file in `config_directory` (it coordinates draive
  processes only) through `claude auth status`; a run only gets a short-lived
  `ClaudeCodeAccessToken` and can not refresh it, so `refresh_margin` has to be longer than the
  longest run. A run whose token is still valid waits for a refresh at most `refresh_timeout` and
  goes on with its token, the `ClaudeCode` scope waits for refreshes still in flight when it exits.
- `ClaudeCodeLoginUnusable`, a `ClaudeCodeUnavailable` raised when a `ClaudeCodeLocalLogin` gives no
  valid token - its credentials file is missing, unreadable or malformed, or its token has expired
  and could not be refreshed - so a caller can tell that logging in again is needed.
- `ClaudeCodeConfig.claude_md` loads the `CLAUDE.md` of the working directory per run without
  enabling the project settings: `ClaudeCodeProjectClaudeMd()` (default `ClaudeCodeNoClaudeMd()`).
  It reads `<directory>/CLAUDE.md` and `<directory>/.claude/CLAUDE.md` only - no ancestors, no
  `CLAUDE.local.md`, no nested files - and the hooks and settings of the project stay off. The
  `CLAUDE.md` of additional directories is excluded unless
  `additional_directories=ClaudeCodeLoadAdditionalClaudeMd()`.
- `ClaudeCodeAccountConnectors` filters connectors: `excluded` (display names, never prefixed with
  `claude.ai `), `excluded_urls` (URL patterns) and `allowed`, `ClaudeCodeAllConnectors()` by
  default or `ClaudeCodeConnectorsAt(urls=...)`, which loads only the connectors at the URLs. An
  allowlist keeps the configured `mcp_servers` (allowed by their names automatically, a run fails
  with `ClaudeCodeUnavailable` when a name has other characters than letters, digits, `_` and `-`)
  and drops every other MCP server, the ones of plugins included. `claudeMdExcludes`,
  `allowedMcpServers` and `deniedMcpServers` of `ClaudeCodeConfig.settings` have to be lists and are
  added to the derived entries instead of replacing them.
- `ClaudeCodeAuthenticationFailed`, a `ClaudeCodeErrored` raised when the API rejected the
  credential of a run (HTTP 401 or an `authentication_error`), so a caller using
  `ClaudeCodeLocalLogin` can tell it and run again. `ClaudeCodeFailure.api_error_type`,
  `ClaudeCodeMCPServer.status` (`ClaudeCodeMCPStatus`) and `.source` are reported from the stream.
- A run with account connectors records the connectors which failed or need an authorization as
  `claude_code.connectors_unavailable` (names only, at most ten) -
  `ClaudeCodeStarted.unavailable_connectors` - and warns only about the ones which failed
  (`ClaudeCodeStarted.failed_connectors`): most connectors which need an authorization were never
  authorized by the account. `ClaudeCodeStarted.failed_mcp_servers` lists the other servers which
  did not connect. Connectors removed by the filters are not reported by Claude Code at all.
- `claude_code_agent(tools=...)` exposes draive tools to Claude Code over MCP, as the `draive` server
  (reserved name, allowed automatically, also next to an allowlist of connectors). Only tools with
  the `response` handling are accepted, a run fails with `ClaudeCodeUnavailable` when that server
  does not connect.
- `ClaudeCodeUnfinished`, a `ClaudeCodeFailed` raised when the output of a run ended without a
  result while no exit of the process is known (it kept running, or a custom runner ended its
  stream).
- A run logs the built-in plugins of Claude Code which the isolating settings do not know (warning)
  and the ones an organization enforces despite being disabled (info), with
  `ClaudeCodeLaunch.leaked_plugins` and `ClaudeCodeLaunch.enforced_plugins`.

### Changed

- **Breaking:** requires haiway 0.66.0. Its State validation and JSON are stricter: a mutable plain
  default (`{}`, `[]`, `set()`) of a State attribute raises `TypeError` when the class is created -
  use `Default(factory=...)` or an immutable value - no string is parsed into a number, and
  `to_json` writes compact, strict JSON. The JSON schema of a State agrees with its validation: a
  `bool` accepts `0`, `1` and the words `true` and `false`, `datetime`, `time` and `timedelta` state
  no `format`, and a closed tuple states `minItems` and `maxItems`.
- **Breaking:** `claude_code_agent` keeps memory only as `sessions` (`ClaudeCodeSessions`), the
  `memory` argument and `AgentMemory` support are removed - a call without sessions is a fresh
  session.
- **Breaking:** `ClaudeCodeConfig.tools` is empty by default, a run has no built-in tools unless
  they are listed (it was `Read`, `Grep` and `Glob`).
- **Breaking:** runs with `Bash` among the `tools` are sandboxed by default -
  `ClaudeCodeConfig.sandbox` is `ClaudeCodeSandbox()`, which confines the commands to the working and
  additional directories with the sandbox of Claude Code (strict: nothing reruns outside of it, a run
  fails when the sandbox can not start - Linux needs bubblewrap, socat and unprivileged user
  namespaces). Commands stay allowed by the `allowed_tools` rules (no auto-approval), network is
  blocked unless `allowed_domains`; `ClaudeCodeSandbox` takes `allowed_writes`, `denied_writes`,
  `denied_reads`, `allowed_reads`, `excluded_commands`, `allowed_domains` and `denied_domains`.
  Use `ClaudeCodeNoSandbox()` within a container which isolates the run, extra keys go to
  `settings["sandbox"]`. Without the sandbox `Bash` can call draive tools directly, so
  `disallowed_tools` rules for them are not enforced against it.
- The `claude` executable is resolved from `binary=`, `CLAUDE_CODE_BINARY` and the `PATH` only.
- The launch helpers (`build_launch`, `claude_arguments`, `claude_environment`, `claude_settings`,
  `inherited_environment`, `verify_connectors`, `verify_allowed_servers` and `verify_claude_md`)
  are no longer module functions of `draive.claude.launch`, they are `ClaudeCodeLaunch.of`,
  `prepare_arguments`, `prepare_environment`, `prepare_settings`, `inherited_environment`,
  `verify_connectors`, `verify_allowed_servers` and `verify_claude_md`. The plugin which an
  organization enforces is no longer hardcoded.
- Runs always exclude the private `CLAUDE.md` of the user (`~/CLAUDE.md`, `~/CLAUDE.local.md`,
  `~/.claude/CLAUDE.md` and `~/.claude/**`) through `claudeMdExcludes`, configured exclusions
  are added to them.
- The text blocks of the agent are streamed apart as paragraphs.
- `ClaudeCodeLaunch.credential` is now a `ClaudeCodeResolvedCredential`, and `ClaudeCodeRunner`
  has a `resolving` (`ClaudeCodeResolving`) of the credential of each run. New exports:
  `ClaudeCodeAccessToken`, `ClaudeCodeResolvedCredential`, `ClaudeCodeResolving` and
  `ClaudeCodeConnectors`.
- `ClaudeCodeMCPServer` keeps the `status` and `source` Claude Code reports, `connected` is derived
  from the status and it is no longer a constructor argument.
- `ClaudeCodeErrored` and `ClaudeCodeUnavailable` are not `final` anymore,
  `ClaudeCodeAuthenticationFailed` and `ClaudeCodeLoginUnusable` extend them.
- `MCP_CONNECTION_NONBLOCKING` is a reserved environment variable of runs, a configuration setting
  it in `env` or `inherit_env` is refused.

### Fixed

- SurrealDB field definitions keep their types under haiway 0.66.0: a `datetime` attribute is a
  `datetime` field and a `bool` one a `bool` field, also within optionals and sequences, taken from
  the annotations as the schema of a State no longer tells them apart. An attribute which can be
  `Missing` is an `option`, as a missing value is left out of the record.
- An MCP tool whose input schema has a closed tuple without `minItems` and `maxItems` loads again:
  the length of the tuple is stated, every position required.

## 0.122.0

### Changed

- Requires haiway 0.65.0. A `TypedDict` attribute now refuses keys it does not declare instead of
  dropping them, and `State.from_mapping`, `State.from_json` and `State.from_json_array` accept
  `ignore_extra=True` to skip unknown keys.
- Dependencies refreshed to their latest releases.
- `ModelToolFunctionSpecification.parameters`, `Tool.parameters` and the `parameters` of
  `CoroutineTool` and `tool` take `ModelToolParameters`: a `ModelToolParametersSpecification` or a
  plain JSON schema mapping, as declared by external tools.

### Fixed

- MCP tool input schemas are translated into `ModelToolParametersSpecification`, they failed to
  convert under haiway 0.65.0: references to `$defs` and `definitions` are inlined (recursive
  definitions anchored as recursive States are), references with sibling keywords and `allOf`
  branches are combined, `oneOf` becomes `anyOf`, `const` a single value `enum`, arguments of root
  `anyOf` branches become arguments (alternatives of a property kept as `anyOf`), objects get an
  explicit `additionalProperties` and keywords which only constrain values further (`default`,
  `minimum`, `pattern`...) are skipped. A tool whose schema can not be translated without changing
  what it accepts is skipped with a warning instead of failing the whole tools fetch. Schemas with keywords outside of
  `ModelToolParametersSpecification` (`title`, `default`, `minimum`, `$defs` and `$ref`, `oneOf`...)
  failed to convert under haiway 0.65.0, and `$defs` were dropped before.
- Gemini skips the thinking config when the thinking budget is zero (0.121.1).

## 0.121.0

### Changed

- Requires haiway 0.64.0.
- `Gemini()` and `Ollama()` provide `DecisionModel` within their default features.
- Gemini requires google-genai 3.6.1, Ollama requires ollama 0.6.3.
- Public re-exports from haiway now include `StateResponse` of `draive.starlette` and
  `draive.fastapi` (haiway 0.64.0), which answers with a `State` or a sequence of them as the JSON
  body.

### Added

- `draive.claude` - a Claude Code agent (`claude_code_agent`) running the `claude` executable as a
  fully managed process, no extra needed (see `docs/guides/ClaudeCode.md`):
    - Authenticates only with the given credential, `ClaudeCodeApiKey` or `ClaudeCodeOAuthToken`
      (`ClaudeCode`), handed over through a file descriptor and kept out of arguments, environment
      and reprs. The user login is out of reach of the isolated process and runs reporting another
      authentication are stopped.
    - `ClaudeCodeConfig` configures the model and effort, permission mode, tools and rules, limits
      (turns, cost, deadline), MCP servers, subagents, plugins, settings, directories, environment
      and compaction. It refuses what can not work when created, `compaction` and credentials are
      sum types. Runs are isolated and read-only by default, the file tools are confined to the
      working and additional directories, `setting_sources` opts into the project `CLAUDE.md`.
    - Streams text and tool use as `ProcessingEvent`s, supports images, failures are the variants
      of `ClaudeCodeFailed` (`ClaudeCodeErrored`, `ClaudeCodeTimedOut`, `ClaudeCodeEnded`,
      `ClaudeCodeOutputTooLarge`) and `ClaudeCodeUnavailable`, MCP servers which fail to connect are
      reported.
    - The `memory` of the agent is either `AgentMemory` of draive or `ClaudeCodeSessions` kept by
      Claude Code in a state directory, resumed also after an interrupted run. Sessions are found
      by the stable identity of the agent.
    - Exports Claude Code telemetry to the collector of draive with the trace of the calling agent
      continued (`ClaudeCodeTelemetry`), and records usage and cost in draive telemetry.
    - A run is stopped with its closed stream, cancellation or deadline - terminated gracefully,
      killed after `stop_grace` - and on Linux what it started with its environment marker is
      stopped too, also a command detached from the process group. POSIX platforms only.
- `Step.route(criteria, *steps, input=)` lets `DecisionModel` pick the step to execute from the
  context (or a custom `StepDecisionInputPreparing`), steps are offered by the `name` and
  `description` of their meta.
- `Step.meta` and `Step.with_meta(...)`, wrapping modifiers (`with_retry`, `with_timeout`,
  `with_fallback`, ...) keep the meta of the step.
- Decision generation: `DecisionGeneration` with `decide`, `check`, `choose`, `select` and `rate`
  answering `DecisionCheck`, `DecisionChoice`, `DecisionSelection` and `DecisionRating` questions
  about a shared input. Options and levels are `DecisionOption`s carrying caller meta, answers
  carry the decided option with backend specific values (i.e. calibrated probabilities) in meta.
  A decision backend has to be bound; `generate_completion_decisions` is the optional
  implementation answering with structured `GenerativeModel` completions (slow and costly, never
  used implicitly). See `docs/guides/BasicDecisionGeneration.md`.
- `DecisionModel` (`draive.models`) - thin decision layer next to `GenerativeModel` holding the
  decision question and answer types and the backend, `DecisionGeneration` is the high level
  facade over it. Providers install `DecisionModel`.
- Gemini Decisions API support (`GeminiDecisionConfig`), Ollama decision models support
  (`OllamaDecisionConfig`) and `draive.systemone` (`draive[systemone]`) for any System One API
  server - TypeSafe Jev, Ollama, gateways or self-hosted OpenJev.
- Decision telemetry: `record_decision_invocation` (attributes `decision.provider`,
  `decision.model`, `decision.questions`, `decision.question_types`) and `record_decision_metrics`
  (counters `decision.questions`, `decision.requests`), recorded by the Gemini, Ollama, System One
  and completion based decision backends.
- `State` types take a `description` class keyword (haiway 0.63.0), e.g.
  `class Answer(State, description="Final answer")`. It is written into the type's JSON schema, so
  structured output, tool arguments and MCP specifications carry it, and it survives `python -OO`.

### Fixed

- Merging many text chunks into `MultimodalContent` is linear instead of quadratic, joining a run of
  texts once. Together with a text fast path of `model_output_blocks` aggregating long output
  streams is up to ~35x faster (32k chunks: 131 ms to 3.7 ms).
- Gemini requests with system instructions no longer fail on the Gemini Developer API, the client
  uses its `v1beta` API version unless a version or Vertex AI is selected explicitly.

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
- MCP: cyclic pagination cursors no longer loop forever, a failing `MCPClients` enter releases
  the clients which already entered, closing releases every client, duplicate client identifiers
  are rejected, tool name collisions across servers are logged, resource links are passed through
  instead of failing the tool call (client and server), the ASGI apps no longer return debug
  tracebacks, the SSE endpoint returns a response on disconnect and clients accept an optional
  `request_timeout` for every request.
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
