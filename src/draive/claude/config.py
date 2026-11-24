from collections.abc import Mapping, Sequence, Set
from os.path import isabs
from typing import Annotated, Final, Literal, Self, final

from haiway import BasicValue, Configuration, Missing, Sensitive, State, Verifier, getenv_str

__all__ = (
    "RUN_MARKER_VARIABLE",
    "ClaudeCodeAccountConnectors",
    "ClaudeCodeAutoCompaction",
    "ClaudeCodeCompaction",
    "ClaudeCodeConfig",
    "ClaudeCodeConnectors",
    "ClaudeCodeNoCompaction",
    "ClaudeCodeNoConnectors",
    "ClaudeCodeTelemetry",
)

# set within the environment of every run, and of everything it starts, to find it again
RUN_MARKER_VARIABLE: Final[str] = "DRAIVE_CLAUDE_RUN"

_RESERVED_ENV_PREFIXES: Final[tuple[str, ...]] = ("CLAUDE", "ANTHROPIC_", "DISABLE_", "OTEL_")
_RESERVED_ENV_NAMES: Final[tuple[str, ...]] = (
    RUN_MARKER_VARIABLE,
    "HOME",
    "TMPDIR",
    "TRACEPARENT",
    "TRACESTATE",
    "DO_NOT_TRACK",
    "NODE_OPTIONS",
)
_MAXIMAL_PERCENT: Final[int] = 100
_MINIMAL_COMPACTION_WINDOW: Final[int] = 100_000


def _positive_integer(
    value: int,
) -> int:
    if value < 1:
        raise ValueError("Value has to be at least 1")

    return value


def _positive_number(
    value: float,
) -> float:
    if value <= 0:
        raise ValueError("Value has to be positive")

    return value


def _not_negative_number(
    value: float,
) -> float:
    if value < 0:
        raise ValueError("Value can not be negative")

    return value


def _percent(
    value: int,
) -> int:
    if not 1 <= value <= _MAXIMAL_PERCENT:
        raise ValueError(f"Percent has to be between 1 and {_MAXIMAL_PERCENT}")

    return value


def _compaction_window(
    value: int,
) -> int:
    if value < _MINIMAL_COMPACTION_WINDOW:
        raise ValueError(f"Window has to be at least {_MINIMAL_COMPACTION_WINDOW} tokens")

    return value


def _absolute(
    value: str,
) -> str:
    if not isabs(value):
        raise ValueError(f"Directory '{value}' has to be absolute")

    return value


def _absolute_directories(
    value: Sequence[str],
) -> Sequence[str]:
    for directory in value:
        _absolute(directory)

    return value


def _unreserved_names(
    value: Sequence[str],
) -> Sequence[str]:
    for name in value:
        if name in _RESERVED_ENV_NAMES or name.startswith(_RESERVED_ENV_PREFIXES):
            raise ValueError(f"Environment variable '{name}' is reserved")

    return value


def _unreserved_environment(
    value: Mapping[str, str],
) -> Mapping[str, str]:
    _unreserved_names(tuple(value))
    return value


def _signals(
    value: Set[Literal["metrics", "logs", "traces"]],
) -> Set[Literal["metrics", "logs", "traces"]]:
    if not value:
        raise ValueError("Telemetry needs at least one signal to export")

    return value


def _protocol(
    value: str | None,
    /,
) -> Literal["grpc", "http/protobuf", "http/json"]:
    match value:
        case None | "":
            return "grpc"

        case "grpc" | "http/protobuf" | "http/json":
            return value

        case _:
            raise ValueError(f"Unsupported OTLP protocol '{value}'")


@final
class ClaudeCodeAutoCompaction(State, serializable=True):
    """Claude Code compacts the context of a session when it grows.

    Attributes
    ----------
    percent
        Share of the context window in percent which triggers compaction, lower than default.
    window
        Context window in tokens used to compute when to compact, at least 100000.
    """

    percent: Annotated[int, Verifier(_percent)] | Missing = Missing
    window: Annotated[int, Verifier(_compaction_window)] | Missing = Missing
    type: Literal["auto"] = "auto"


@final
class ClaudeCodeNoCompaction(State, serializable=True):
    """Claude Code does not compact the context of a session."""

    type: Literal["none"] = "none"


type ClaudeCodeCompaction = ClaudeCodeAutoCompaction | ClaudeCodeNoCompaction


@final
class ClaudeCodeNoConnectors(State, serializable=True):
    """Runs do not use the connectors of the claude.ai account, only the configured MCP servers."""

    type: Literal["none"] = "none"


@final
class ClaudeCodeAccountConnectors(State, serializable=True):
    """Runs use the connectors of the claude.ai account - its cloud MCP servers.

    Connectors are fetched at the start of every run and their tools are named
    ``mcp__claude_ai_<Name>__<tool>``, allow them with ``allowed_tools`` of the configuration -
    nothing allows them otherwise. They need a credential declaring the ``user:mcp_servers``
    scope, which is only a ``ClaudeCodeLocalLogin``. MCP servers of the configuration are still
    loaded.

    Attributes
    ----------
    wait
        Whether Claude Code waits (for a limited time, which it caps) for the connectors to
        connect before the model starts. Without it the tools of connectors which are not
        connected yet are not seen by the model.
    """

    wait: bool = True
    type: Literal["account"] = "account"


type ClaudeCodeConnectors = ClaudeCodeNoConnectors | ClaudeCodeAccountConnectors


@final
class ClaudeCodeTelemetry(State, serializable=True):
    """OpenTelemetry export of Claude Code, to the system collecting the telemetry of draive.

    Claude Code exports its metrics, events and - as a beta - traces over OTLP. A run continues the
    trace of the agent executing it, so everything lands under the span of the agent within the
    same trace.

    Attributes
    ----------
    endpoint
        OTLP collector endpoint, Claude Code defaults to a local collector without it.
    protocol
        OTLP protocol, ``grpc`` matches the exporters draive configures.
    headers
        OTLP headers as ``key=value`` pairs separated with commas, i.e. credentials.
    signals
        What is exported: ``metrics`` (``claude_code.token.usage``, ``claude_code.cost.usage``,
        ...), ``logs`` - the events (``claude_code.api_request``, ...) - and ``traces``, a beta of
        Claude Code (``claude_code.interaction``, ...).
    export_interval_ms
        Export interval of all signals, short as runs end before the defaults of Claude Code.
    resource_attributes
        Resource attributes of everything exported by the run, ``service.name`` is
        ``claude-code`` unless given.
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
        **fields: object,
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
        return cls(
            endpoint=getenv_str("OTEL_EXPORTER_OTLP_ENDPOINT") or Missing,
            protocol=_protocol(getenv_str("OTEL_EXPORTER_OTLP_PROTOCOL")),
            headers=getenv_str("OTEL_EXPORTER_OTLP_HEADERS") or Missing,
        ).updating(**fields)

    endpoint: str | Missing = Missing
    protocol: Literal["grpc", "http/protobuf", "http/json"] = "grpc"
    headers: Annotated[str, Sensitive()] | Missing = Missing
    signals: Annotated[Set[Literal["metrics", "logs", "traces"]], Verifier(_signals)] = frozenset(
        ("metrics", "logs")
    )
    export_interval_ms: Annotated[int, Verifier(_positive_integer)] = 2000
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
        Limit of the run duration in seconds, the process is terminated after it.
    stop_grace
        Time in seconds a terminated process has to exit before it is killed.
    stream_deltas
        Stream the text while it is generated instead of complete text blocks.
    system_prompt
        Whether the instructions ``append`` to the system prompt of Claude Code, which keeps
        its tool guidance, or ``replace`` it.
    mcp_servers
        MCP servers by name, i.e. ``{"command": "npx", "args": [...]}`` or
        ``{"type": "http", "url": ...}``, nothing else is loaded unless ``connectors`` are used.
    connectors
        Whether runs use the connectors of the claude.ai account, none by default.
    subagents
        Subagents by name, with ``description`` and ``prompt`` at least.
    plugin_directories
        Absolute directories of plugins to load, skills and commands are disabled without them.
    settings
        Additional settings of Claude Code, merged over the isolating ones.
    setting_sources
        Settings Claude Code loads besides the given ones, none by default. ``project`` loads
        the ``CLAUDE.md`` and the ``.claude`` settings of the working directory - use it when
        the agent works within a project which has its own rules, ``user`` and ``local`` load
        the ones of the private home and of the directory. Project settings can add hooks and
        MCP servers, which is what the isolation keeps out otherwise.
    directory
        Absolute working directory, a temporary one removed after each run when not set.
    additional_directories
        Absolute directories Claude Code can access besides the working one.
    env
        Environment variables of the process, ``CLAUDE*``, ``ANTHROPIC_*``, ``OTEL_*`` and
        similar are reserved.
    inherit_env
        Names of environment variables taken from the parent process, besides ``PATH``,
        locale, proxy and certificate ones which are always inherited.
    compaction
        Whether Claude Code compacts the context of a session when it grows, and when.
    telemetry
        Export of the Claude Code telemetry, none when not set.
    """

    model: str = "sonnet"
    effort: Literal["low", "medium", "high", "xhigh", "max"] | Missing = Missing
    permission_mode: Literal["dontAsk", "acceptEdits", "plan", "bypassPermissions"] = "dontAsk"
    tools: Sequence[str] = ("Read", "Grep", "Glob")
    allowed_tools: Sequence[str] | Missing = Missing
    disallowed_tools: Sequence[str] = ()
    max_turns: Annotated[int, Verifier(_positive_integer)] | Missing = Missing
    max_budget_usd: Annotated[float, Verifier(_positive_number)] | Missing = Missing
    timeout: Annotated[float, Verifier(_positive_number)] = 1800.0
    stop_grace: Annotated[float, Verifier(_not_negative_number)] = 10.0
    stream_deltas: bool = False
    system_prompt: Literal["append", "replace"] = "append"
    mcp_servers: Mapping[str, Mapping[str, BasicValue]] = {}
    connectors: ClaudeCodeConnectors = ClaudeCodeNoConnectors()
    subagents: Mapping[str, Mapping[str, BasicValue]] = {}
    plugin_directories: Annotated[Sequence[str], Verifier(_absolute_directories)] = ()
    settings: Mapping[str, BasicValue] = {}
    setting_sources: Sequence[Literal["user", "project", "local"]] = ()
    directory: Annotated[str, Verifier(_absolute)] | Missing = Missing
    additional_directories: Annotated[Sequence[str], Verifier(_absolute_directories)] = ()
    env: Annotated[Mapping[str, str], Verifier(_unreserved_environment)] = {}
    inherit_env: Annotated[Sequence[str], Verifier(_unreserved_names)] = ()
    compaction: ClaudeCodeAutoCompaction | ClaudeCodeNoCompaction = ClaudeCodeAutoCompaction()
    telemetry: ClaudeCodeTelemetry | Missing = Missing
