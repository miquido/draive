import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal, cast, final

from haiway import BasicValue, Default, State

__all__ = (
    "ClaudeCodeDelta",
    "ClaudeCodeEvent",
    "ClaudeCodeFailure",
    "ClaudeCodeMCPServer",
    "ClaudeCodeMCPStatus",
    "ClaudeCodeResult",
    "ClaudeCodeStarted",
    "ClaudeCodeSuccess",
    "ClaudeCodeText",
    "ClaudeCodeToolResult",
    "ClaudeCodeToolUse",
    "ClaudeCodeUsage",
    "parse_events",
)


_CONNECTOR_SOURCE: Final[str] = "claudeai"
_UNAUTHORIZED: Final[int] = 401
_UNAVAILABLE_STATUSES: Final[Sequence[str]] = ("failed", "needs-auth")


type ClaudeCodeMCPStatus = Literal[
    "connected", "failed", "needs-auth", "pending", "disabled", "unknown"
]


@final
class ClaudeCodeMCPServer(State):
    """MCP server reported by Claude Code, whether it was connected or not.

    Attributes
    ----------
    name
        Name of the server, connectors of the account are named ``claude.ai <Name>``.
    status
        Status reported by Claude Code, ``unknown`` for one this version does not know and none
        when it reports no status.
    source
        Where the server comes from when reported, connectors of the account are ``claudeai``.
    """

    name: str
    status: ClaudeCodeMCPStatus | None = None
    source: str | None = None

    @property
    def connected(self) -> bool:
        """Whether the server is connected, also when its status is not reported."""
        return self.status in (None, "connected")


@final
class ClaudeCodeStarted(State):
    """The run reported its session, ``system`` ``init`` of the stream."""

    session_id: str
    model: str | None = None
    version: str | None = None
    permission_mode: str | None = None
    api_key_source: str | None = None
    tools: Sequence[str] = ()
    mcp_servers: Sequence[ClaudeCodeMCPServer] = ()
    plugins: Sequence[str] = ()

    @property
    def failed_mcp_servers(self) -> Sequence[str]:
        """Names of the configured MCP servers which are not connected, connectors excluded."""
        return tuple(
            server.name
            for server in self.mcp_servers
            if not server.connected and server.source != _CONNECTOR_SOURCE
        )

    @property
    def unavailable_connectors(self) -> Sequence[str]:
        """Names of the connectors of the account which failed or need an authorization."""
        return tuple(
            server.name
            for server in self.mcp_servers
            if server.source == _CONNECTOR_SOURCE and server.status in _UNAVAILABLE_STATUSES
        )

    @property
    def failed_connectors(self) -> Sequence[str]:
        """Names of the connectors of the account which failed to connect - the ones which need
        an authorization are not among them, most were never authorized by the account."""
        return tuple(
            server.name
            for server in self.mcp_servers
            if server.source == _CONNECTOR_SOURCE and server.status == "failed"
        )


@final
class ClaudeCodeText(State):
    """Complete text block of an assistant message."""

    text: str
    parent: str | None = None  # tool use of the main agent the subagent text belongs to


@final
class ClaudeCodeDelta(State):
    """Text fragment of the message being generated, a preview of its complete text block."""

    text: str


@final
class ClaudeCodeToolUse(State):
    """Request of the agent to use a tool."""

    identifier: str
    tool: str
    arguments: Mapping[str, BasicValue] = Default(factory=dict[str, BasicValue])
    parent: str | None = None


@final
class ClaudeCodeToolResult(State):
    """Result of the tool use reported back to the agent."""

    identifier: str
    content: str
    failed: bool = False
    parent: str | None = None


@final
class ClaudeCodeUsage(State):
    """Tokens and cost consumed by a model within the run."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None


@final
class ClaudeCodeSuccess(State):
    """The run ended with an answer.

    Attributes
    ----------
    text
        Final text of the run.
    """

    text: str = ""


@final
class ClaudeCodeFailure(State):
    """The run ended with an error - an API failure, a reached limit, ...

    Claude Code still reports ``subtype`` ``success`` for some of them.

    Attributes
    ----------
    reason
        Description of the error, the text of the result or its subtype.
    api_error_status
        HTTP status of the API failure which ended the run, when it was one.
    terminal_reason
        Reason reported by Claude Code for ending the run, i.e. ``api_error``.
    api_error_type
        Type of the API failure which ended the run, i.e. ``authentication_error``.
    startup_failure
        Reason reported by Claude Code for not starting the run at all, i.e.
        ``sandbox_required_unavailable``.
    """

    reason: str
    api_error_status: int | None = None
    terminal_reason: str | None = None
    api_error_type: str | None = None
    startup_failure: str | None = None

    @property
    def authentication_failed(self) -> bool:
        """Whether the API rejected the credential of the run."""
        return (
            self.api_error_type == "authentication_error" or self.api_error_status == _UNAUTHORIZED
        )


@final
class ClaudeCodeResult(State):
    """Outcome of the run and what it consumed, the last event of a stream.

    Attributes
    ----------
    outcome
        The answer or the error the run ended with.
    """

    outcome: ClaudeCodeSuccess | ClaudeCodeFailure
    session_id: str | None = None
    turns: int = 0
    duration_ms: int = 0
    cost_usd: float | None = None
    permission_denials: int = 0
    usage: Sequence[ClaudeCodeUsage] = ()

    @property
    def failed(self) -> bool:
        return isinstance(self.outcome, ClaudeCodeFailure)


type ClaudeCodeEvent = (
    ClaudeCodeStarted
    | ClaudeCodeText
    | ClaudeCodeDelta
    | ClaudeCodeToolUse
    | ClaudeCodeToolResult
    | ClaudeCodeResult
)


def parse_events(
    line: bytes | str,
    /,
) -> Sequence[ClaudeCodeEvent]:
    """Parse a line of the Claude Code output stream into the events it carries.

    Lines which are not JSON objects or carry anything the run does not need are ignored,
    one assistant message can carry several events.

    Parameters
    ----------
    line
        Single line of the ``stream-json`` output.

    Returns
    -------
    Sequence[ClaudeCodeEvent]
        Events in the order of the line, empty when there is nothing of interest.
    """
    try:
        payload: Mapping[str, Any] | None = _mapping(json.loads(line))

    except ValueError:
        return ()

    if payload is None:
        return ()

    parser = _PARSERS.get(payload.get("type"))
    return parser(payload) if parser is not None else ()


def _started(
    payload: Mapping[str, Any],
    /,
) -> Sequence[ClaudeCodeEvent]:
    session_id: str | None = _string(payload.get("session_id"))
    if payload.get("subtype") != "init" or session_id is None:
        return ()

    return (
        ClaudeCodeStarted(
            session_id=session_id,
            model=_string(payload.get("model")),
            version=_string(payload.get("claude_code_version")),
            permission_mode=_string(payload.get("permissionMode")),
            api_key_source=_string(payload.get("apiKeySource")),
            tools=tuple(item for item in _sequence(payload.get("tools")) if isinstance(item, str)),
            mcp_servers=_mcp_servers(payload.get("mcp_servers")),
            plugins=_names(payload.get("plugins")),
        ),
    )


def _assistant(
    payload: Mapping[str, Any],
    /,
) -> Sequence[ClaudeCodeEvent]:
    if payload.get("error") or payload.get("is_api_error_message"):
        return ()  # a synthetic message describing the failure, the result reports it

    message: Mapping[str, Any] | None = _mapping(payload.get("message"))
    if message is None:
        return ()

    parent: str | None = _string(payload.get("parent_tool_use_id"))
    events: list[ClaudeCodeEvent] = []
    for item in _sequence(message.get("content")):
        block: Mapping[str, Any] | None = _mapping(item)
        if block is None:
            continue

        block_type: str | None = _string(block.get("type"))
        if block_type == "text":
            text: str | None = _string(block.get("text"))
            # also a blank block, its deltas were streamed already and have to be matched
            if text:
                events.append(ClaudeCodeText(text=text, parent=parent))

        elif block_type == "tool_use":
            identifier: str | None = _string(block.get("id"))
            tool: str | None = _string(block.get("name"))
            if identifier is not None and tool is not None:
                events.append(
                    ClaudeCodeToolUse(
                        identifier=identifier,
                        tool=tool,
                        arguments=_mapping(block.get("input")) or {},
                        parent=parent,
                    )
                )

        # thinking and anything else stays within Claude Code

    return tuple(events)


def _tool_results(
    payload: Mapping[str, Any],
    /,
) -> Sequence[ClaudeCodeEvent]:
    message: Mapping[str, Any] | None = _mapping(payload.get("message"))
    if message is None:
        return ()

    parent: str | None = _string(payload.get("parent_tool_use_id"))
    events: list[ClaudeCodeEvent] = []
    for item in _sequence(message.get("content")):
        block: Mapping[str, Any] | None = _mapping(item)
        if block is None or block.get("type") != "tool_result":
            continue

        identifier: str | None = _string(block.get("tool_use_id"))
        if identifier is not None:
            events.append(
                ClaudeCodeToolResult(
                    identifier=identifier,
                    content=_tool_content(block.get("content")),
                    failed=block.get("is_error") is True,
                    parent=parent,
                )
            )

    return tuple(events)


def _tool_content(
    content: Any,
    /,
) -> str:
    if isinstance(content, str):
        return content

    texts: list[str] = []
    for item in _sequence(content):
        block: Mapping[str, Any] | None = _mapping(item)
        if block is not None and (text := _string(block.get("text"))) is not None:
            texts.append(text)

    return "\n".join(texts)


def _delta(
    payload: Mapping[str, Any],
    /,
) -> Sequence[ClaudeCodeEvent]:
    if payload.get("parent_tool_use_id") is not None:
        return ()  # only the main agent is previewed

    event: Mapping[str, Any] | None = _mapping(payload.get("event"))
    if event is None or event.get("type") != "content_block_delta":
        return ()

    delta: Mapping[str, Any] | None = _mapping(event.get("delta"))
    if delta is None or delta.get("type") != "text_delta":
        return ()

    text: str | None = _string(delta.get("text"))
    return (ClaudeCodeDelta(text=text),) if text else ()


def _result_events(
    payload: Mapping[str, Any],
    /,
) -> Sequence[ClaudeCodeEvent]:
    return (_result(payload),)


def _result(
    payload: Mapping[str, Any],
    /,
) -> ClaudeCodeResult:
    subtype: str | None = _string(payload.get("subtype"))
    status: Any = payload.get("api_error_status")
    text: str = _string(payload.get("result")) or ""
    outcome: ClaudeCodeSuccess | ClaudeCodeFailure
    # Claude Code reports `success` even for a turn which failed with an API error
    if payload.get("is_error") is True or (subtype or "success") != "success":
        # a run which failed before its first turn describes itself only by its errors
        errors: str = "; ".join(
            error for error in _sequence(payload.get("errors")) if isinstance(error, str)
        )
        outcome = ClaudeCodeFailure(
            reason=text or errors or subtype or "the run failed",
            api_error_status=status if isinstance(status, int) and status else None,
            terminal_reason=_string(payload.get("terminal_reason")),
            api_error_type=_string(payload.get("api_error_type")),
            startup_failure=_string(payload.get("startup_failure_reason")),
        )

    else:
        outcome = ClaudeCodeSuccess(text=text)

    return ClaudeCodeResult(
        outcome=outcome,
        session_id=_string(payload.get("session_id")),
        turns=_integer(payload.get("num_turns")),
        duration_ms=_integer(payload.get("duration_ms")),
        cost_usd=_number(payload.get("total_cost_usd")),
        permission_denials=len(_sequence(payload.get("permission_denials"))),
        usage=_usage(payload.get("modelUsage")),
    )


def _usage(
    usage: Any,
    /,
) -> Sequence[ClaudeCodeUsage]:
    models: Mapping[str, Any] | None = _mapping(usage)
    if models is None:
        return ()

    entries: list[ClaudeCodeUsage] = []
    for model, item in models.items():
        values: Mapping[str, Any] | None = _mapping(item)
        if values is None:
            continue

        entries.append(
            ClaudeCodeUsage(
                model=model,
                input_tokens=_integer(values.get("inputTokens")),
                output_tokens=_integer(values.get("outputTokens")),
                cache_read_tokens=_integer(values.get("cacheReadInputTokens")),
                cache_creation_tokens=_integer(values.get("cacheCreationInputTokens")),
                cost_usd=_number(values.get("costUSD")),
            )
        )

    return tuple(entries)


def _names(
    items: Any,
    /,
) -> Sequence[str]:
    names: list[str] = []
    for item in _sequence(items):
        entry: Mapping[str, Any] | None = _mapping(item)
        if entry is not None and (name := _string(entry.get("name"))) is not None:
            names.append(name)

    return tuple(names)


def _mcp_servers(
    items: Any,
    /,
) -> Sequence[ClaudeCodeMCPServer]:
    servers: list[ClaudeCodeMCPServer] = []
    for item in _sequence(items):
        entry: Mapping[str, Any] | None = _mapping(item)
        if entry is not None and (name := _string(entry.get("name"))) is not None:
            # a status is reported by the versions which connect the servers on start
            servers.append(
                ClaudeCodeMCPServer(
                    name=name,
                    status=_mcp_status(entry.get("status")),
                    source=_string(entry.get("source")),
                )
            )

    return tuple(servers)


def _mcp_status(
    value: Any,
    /,
) -> ClaudeCodeMCPStatus | None:
    match _string(value):
        case None:
            return None

        case "connected" | "failed" | "needs-auth" | "pending" | "disabled" as status:
            return status

        case _:
            return "unknown"


def _mapping(
    value: Any,
    /,
) -> Mapping[str, Any] | None:
    return cast(Mapping[str, Any], value) if isinstance(value, dict) else None


def _sequence(
    value: Any,
    /,
) -> Sequence[Any]:
    return cast(Sequence[Any], value) if isinstance(value, list) else ()


def _string(
    value: Any,
    /,
) -> str | None:
    return value if isinstance(value, str) else None


def _integer(
    value: Any,
    /,
) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number(
    value: Any,
    /,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None

    return float(value)


_PARSERS: Final[Mapping[Any, Callable[[Mapping[str, Any]], Sequence[ClaudeCodeEvent]]]] = {
    "system": _started,
    "assistant": _assistant,
    "user": _tool_results,
    "stream_event": _delta,
    "result": _result_events,
}
