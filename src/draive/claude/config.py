from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, Self, final

from haiway import BasicValue, Configuration, Missing, Sensitive, State, getenv_str

__all__ = (
    "ClaudeCodeConfig",
    "ClaudeCodeTelemetry",
)


@final
class ClaudeCodeTelemetry(State):
    """OpenTelemetry export of Claude Code, to the system collecting the telemetry of draive.

    Claude Code exports its metrics, events and - as a beta - traces over OTLP. A run
    continues the trace of the agent executing it, so everything lands under the span of the
    agent within the same trace.

    Attributes
    ----------
    endpoint
        OTLP collector endpoint, Claude Code defaults to a local collector without it.
    protocol
        OTLP protocol, ``grpc`` matches the exporters draive configures.
    headers
        OTLP headers as ``key=value`` pairs separated with commas, i.e. credentials.
    metrics
        Export metrics (``claude_code.token.usage``, ``claude_code.cost.usage``, ...).
    logs
        Export events (``claude_code.api_request``, ``claude_code.tool_result``, ...).
    traces
        Export traces, a beta of Claude Code (``claude_code.interaction``, ...).
    metrics_interval_ms
        Metrics export interval, short as runs end before the default 60 seconds.
    logs_interval_ms
        Events export interval.
    traces_interval_ms
        Traces export interval.
    service_name
        Name of the service, set within the resource attributes.
    resource_attributes
        Additional resource attributes of everything exported by the run.
    log_prompts
        Include the content of prompts and assistant responses.
    log_tool_details
        Include tool arguments such as commands and file paths.
    log_tool_content
        Include tool inputs and outputs on trace spans.
    propagate_trace
        Continue the trace of the executing agent, through ``TRACEPARENT``.
    """

    @classmethod
    def from_env(
        cls,
        **fields: bool | int | str | Mapping[str, str],
    ) -> Self:
        """Prepare telemetry using the standard ``OTEL_EXPORTER_OTLP_*`` environment variables.

        Parameters
        ----------
        **fields
            Any other field of the telemetry.

        Returns
        -------
        Self
            Telemetry exporting to the collector the process itself is configured with.
        """
        resolved: dict[str, object] = {}
        if endpoint := getenv_str("OTEL_EXPORTER_OTLP_ENDPOINT"):
            resolved["endpoint"] = endpoint

        if protocol := getenv_str("OTEL_EXPORTER_OTLP_PROTOCOL"):
            resolved["protocol"] = protocol

        if headers := getenv_str("OTEL_EXPORTER_OTLP_HEADERS"):
            resolved["headers"] = headers

        return cls(**{**resolved, **fields})  # pyright: ignore[reportArgumentType]

    endpoint: str | Missing = Missing
    protocol: Literal["grpc", "http/protobuf", "http/json"] = "grpc"
    headers: Annotated[str, Sensitive()] | Missing = Missing
    metrics: bool = True
    logs: bool = True
    traces: bool = False
    metrics_interval_ms: int = 5000
    logs_interval_ms: int = 2000
    traces_interval_ms: int = 2000
    service_name: str = "claude-code"
    resource_attributes: Mapping[str, str] = {}
    log_prompts: bool = False
    log_tool_details: bool = False
    log_tool_content: bool = False
    propagate_trace: bool = True


@final
class ClaudeCodeConfig(Configuration):
    """Configuration of Claude Code runs.

    Runs are isolated by default - no user, project or local settings, no auto memory,
    plugins or skills of the installation, only what is configured here. Claude Code
    is allowed to read files and nothing else.

    Attributes
    ----------
    model
        Model alias (``sonnet``, ``opus``, ``haiku``) or a full model name.
    effort
        Reasoning effort, the default of the model when not set.
    permission_mode
        ``dontAsk`` denies everything which is not allowed, ``acceptEdits`` also accepts file
        edits, ``plan`` only plans, ``bypassPermissions`` allows every tool - use it only
        within an environment which is safe to be modified.
    tools
        Built-in tools available to Claude Code, none when empty.
    allowed_tools
        Tools allowed without a permission, rules such as ``Bash(git status:*)`` narrow a tool
        down. When not set the available ``tools`` are allowed, the ones working with files
        (``Read``, ``Grep``, ``Glob``, ``Edit``, ``Write``) within the working and the additional
        directories only, which includes symlinks and ``..`` leading out of them. Other tools,
        i.e. ``Bash``, are not confined to the directories.
    disallowed_tools
        Tools and rules which are always denied.
    max_turns
        Limit of the agentic turns of a run.
    max_budget_usd
        Limit of the cost of a run.
    timeout
        Limit of the run duration in seconds, the process is killed after it.
    stop_grace
        Time in seconds a stopped process has to exit before it is killed.
    stream_deltas
        Stream the text while it is generated instead of complete text blocks.
    system_prompt
        Whether the instructions ``append`` to the system prompt of Claude Code, which keeps
        its tool guidance, or ``replace`` it.
    mcp_servers
        MCP servers by name, i.e. ``{"command": "npx", "args": [...]}`` or
        ``{"type": "http", "url": ...}``, nothing else is loaded.
    subagents
        Subagents by name, with ``description`` and ``prompt`` at least.
    plugin_directories
        Directories of plugins to load, skills and commands are disabled without them.
    settings
        Additional settings of Claude Code, merged over the isolating ones.
    setting_sources
        Settings Claude Code loads besides the given ones, none by default. ``project`` loads
        the ``CLAUDE.md`` and the ``.claude`` settings of the working directory - use it when
        the agent works within a project which has its own rules, ``user`` and ``local`` load
        the ones of the private home and of the directory. Project settings can add hooks and
        MCP servers, which is what the isolation keeps out otherwise.
    directory
        Working directory, a temporary one removed after each run when not set.
    additional_directories
        Directories Claude Code can access besides the working one.
    env
        Environment variables of the process, ``CLAUDE*``, ``ANTHROPIC_*``, ``OTEL_*`` and
        similar are reserved.
    inherit_env
        Names of environment variables taken from the parent process, besides ``PATH``,
        locale, proxy and certificate ones which are always inherited.
    persistence
        ``ephemeral`` runs leave nothing behind, ``resumable`` keeps Claude Code sessions in
        ``state_directory`` and resumes the one of a thread, instead of ``AgentMemory``.
    state_directory
        Directory with the sessions and the home of resumable runs, also holds the working
        directory when it is not set.
    auto_compact
        Whether Claude Code compacts the context of a session when it grows.
    auto_compact_percent
        Share of the context window in percent which triggers compaction, lower than default.
    auto_compact_window
        Context window in tokens used to compute when to compact, at least 100000.
    telemetry
        Export of the Claude Code telemetry, none when not set.
    """

    model: str = "sonnet"
    effort: Literal["low", "medium", "high", "xhigh", "max"] | Missing = Missing
    permission_mode: Literal["dontAsk", "acceptEdits", "plan", "bypassPermissions"] = "dontAsk"
    tools: Sequence[str] = ("Read", "Grep", "Glob")
    allowed_tools: Sequence[str] | Missing = Missing
    disallowed_tools: Sequence[str] = ()
    max_turns: int | Missing = Missing
    max_budget_usd: float | Missing = Missing
    timeout: float = 1800.0
    stop_grace: float = 10.0
    stream_deltas: bool = False
    system_prompt: Literal["append", "replace"] = "append"
    mcp_servers: Mapping[str, Mapping[str, BasicValue]] = {}
    subagents: Mapping[str, Mapping[str, BasicValue]] = {}
    plugin_directories: Sequence[str] = ()
    settings: Mapping[str, BasicValue] = {}
    setting_sources: Sequence[Literal["user", "project", "local"]] = ()
    directory: str | Missing = Missing
    additional_directories: Sequence[str] = ()
    env: Mapping[str, str] = {}
    inherit_env: Sequence[str] = ()
    persistence: Literal["ephemeral", "resumable"] = "ephemeral"
    state_directory: str | Missing = Missing
    auto_compact: bool = True
    auto_compact_percent: int | Missing = Missing
    auto_compact_window: int | Missing = Missing
    telemetry: ClaudeCodeTelemetry | Missing = Missing
