# Changelog

## Unreleased

### Changed

- Requires Python 3.15 and haiway 0.62.
- Resource templates use haiway `URITemplate` for expansion, matching and verification.
  `ResourceReferenceTemplate.template_uri` is a `URITemplate` (serialized as
  `{"template": "..."}`), `@resource(uri_template=...)` accepts a string or a `URITemplate`.
  URIs carrying undeclared query parameters (for templates without a query) or encoded
  separators within host variables are refused, a missing required variable fails the
  expansion, aliased variables expand correctly.
- Public re-exports from haiway now include `URITemplate`, `DefaultEnv`, `DefaultFactory`,
  `Directory`, `PaginationToken`, `Paths`, `ScopeKind` and `parse_bool`.
- Mistral reads `MISTRAL_ENDPOINT` as an alias for `MISTRAL_SERVER_URL`.
- SurrealDB 3.x or newer is required (verified with 3.2 and 4.0).

### Added

- Telemetry: provider failure/rate-limit metrics, embedding token usage, tool invocations,
  guardrail verdicts, evaluator suite case errors and durations, Qdrant and SurrealDB
  operation metrics, per-root AWS CloudWatch traces.
- `ConversationException`; iteration limits for agents, steps and conversation completion
  (`max_iterations`) raising typed exceptions.
- `docs/guides/Resources.md`.

### Fixed

- Provider adapters translate SDK failures into typed model exceptions consistently
  (OpenAI, Anthropic, Mistral, Gemini, Ollama, Cohere, vLLM, Bedrock).
- Evaluation: invalid score bounds and empty suites are rejected, suite case errors become
  failed results, evaluator signature errors surface.
- Postgres, Qdrant and SurrealDB integrations: transactions, batching, MMR overfetch,
  quoting and error types.
- Unsupported multimodal elements and non-multimodal tool results raise typed errors.
