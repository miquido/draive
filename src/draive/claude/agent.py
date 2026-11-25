import json
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Final, final
from urllib.parse import quote

from haiway import BasicValue, Meta, MetaValues, Missing, State, ctx

from draive.agents import Agent, AgentIdentity, AgentMessage, AgentThread
from draive.claude.config import (
    ClaudeCodeAccountConnectors,
    ClaudeCodeConfig,
    ClaudeCodeConnectors,
    ClaudeCodeNoConnectors,
)
from draive.claude.credentials import ClaudeCodeResolvedCredential
from draive.claude.events import (
    ClaudeCodeDelta,
    ClaudeCodeEvent,
    ClaudeCodeFailure,
    ClaudeCodeResult,
    ClaudeCodeStarted,
    ClaudeCodeSuccess,
    ClaudeCodeText,
    ClaudeCodeToolResult,
    ClaudeCodeToolUse,
)
from draive.claude.launch import ClaudeCodeLaunch
from draive.claude.prompt import prompt_blocks
from draive.claude.sessions import (
    ClaudeCodeNewSession,
    ClaudeCodeNoSession,
    ClaudeCodeResumedSession,
    ClaudeCodeSession,
    ClaudeCodeSessions,
)
from draive.claude.tools import TOOLS_SERVER_KEY, ClaudeCodeToolsBridge
from draive.claude.types import (
    ClaudeCodeErrored,
    ClaudeCodeException,
    ClaudeCodeRunner,
    ClaudeCodeUnavailable,
    ClaudeCodeUnfinished,
)
from draive.claude.workspace import ClaudeCodeWorkspace
from draive.models import record_usage_metrics
from draive.multimodal import (
    MultimodalContentPart,
    Template,
    TemplatesRepository,
    TextContent,
)
from draive.tools import Tool, Toolbox
from draive.utils import ProcessingEvent

__all__ = ("claude_code_agent",)

_PROVIDER: Final[str] = "claude_code"
# connectors named by the record and the warning, the rest is counted
_REPORTED_CONNECTORS: Final[int] = 10

type _Chunk = MultimodalContentPart | ProcessingEvent

_PARAGRAPH: Final[str] = "\n\n"  # separates the text blocks of the main agent
_LINE_BREAKS: Final[str] = "\r\n"


@final
class _Streamed(State):
    """What was streamed of a run so far, the only source of its text, session and result."""

    previewed: str = ""  # streamed deltas which their complete block is not to repeat
    text: str = ""  # text of the main agent exactly as it was streamed
    separated: bool = False  # a block of the text ended, the next one starts a paragraph
    started: str | None = None
    result: ClaudeCodeResult | None = None


def claude_code_agent(
    agent: AgentIdentity | str,
    *,
    description: str = "",
    instructions: Template | str = "",
    sessions: ClaudeCodeSessions | None = None,
    config: ClaudeCodeConfig | None = None,
    tools: Toolbox | Iterable[Tool] = Toolbox.empty,
    meta: Meta | MetaValues | None = None,
) -> Agent:
    """Create an agent which runs Claude Code, fully managed.

    Every call runs the ``claude`` executable as a process isolated from the user setup of
    Claude Code, streams its work and stops it whatever way the call ends. The text of the
    agent is streamed as text parts, its tool use and results as ``ProcessingEvent``s named
    ``claude_code.tool_use``, ``claude_code.tool_result`` and ``claude_code.result``.

    With ``sessions`` Claude Code keeps the session of each thread in the state directory and
    resumes it with the whole context it had - tool use, compaction and prompt caching. Without
    it every call is a fresh session and nothing is kept.

    The call is an async generator, close it when it is left early, i.e. with ``ctx.closing``.

    Parameters
    ----------
    agent
        Identity of the agent, or its name. The identity made of a name is stable, so agents
        of the same name share the sessions of their threads, also after a restart.
    description
        Description of the agent, ignored when an identity is given.
    instructions
        Instructions of the agent, appended to the system prompt of Claude Code, or replacing
        it according to the configuration.
    sessions
        Sessions of the threads kept by Claude Code, every call is a fresh session when not set.
    config
        Configuration of the runs, ``ClaudeCodeConfig`` of the scope when not set.
    tools
        Draive tools exposed to every run as the MCP server ``draive``, they appear as
        ``mcp__draive__<tool>`` within the tool events and are allowed automatically. Distinct
        from ``ClaudeCodeConfig.tools``, which selects the built-in tools of Claude Code. They run
        within the process and the context of the agent, only the tools with ``response`` handling
        are supported. Requires the ``mcp`` extra.
    meta
        Metadata of the identity, ignored when an identity is given.

    Returns
    -------
    Agent
        Agent running Claude Code, it requires ``ClaudeCode`` entered within the scope.

    Raises
    ------
    ClaudeCodeUnavailable
        When called without the ``ClaudeCode`` disposable, or the executable can not be started.
    ClaudeCodeFailed
        When the run failed, timed out or reported an error.
    ValueError
        When the message or the working directory is not supported, or a tool can not be exposed
        to Claude Code - its handling is not ``response`` or its name is not accepted.
    """
    toolbox: Toolbox = Toolbox.of(tools)
    if toolbox.tools:
        ClaudeCodeToolsBridge.verify(toolbox)

    identity: AgentIdentity = (
        # stable, as the sessions are found by it after a restart
        AgentIdentity.of(
            f"agent://claude-code/{quote(agent, safe='')}",
            name=agent,
            description=description,
            meta=meta,
        )
        if isinstance(agent, str)
        else agent
    )

    async def executing(
        message: AgentMessage,
    ) -> AsyncGenerator[_Chunk]:
        async with ctx.scope("agent.claude_code"):
            thread: AgentThread = _current_thread()
            resolved: ClaudeCodeConfig = config or ctx.state(ClaudeCodeConfig)
            runner: ClaudeCodeRunner = ctx.state(ClaudeCodeRunner)
            text: str = await _resolved(instructions)
            async with (
                _serving(toolbox) as tools_server,
                ClaudeCodeWorkspace.prepare(resolved, sessions) as workspace,
            ):
                if "project" in resolved.setting_sources and await workspace.reaches_user_agents():
                    ctx.log_warning(
                        "Claude Code loads the subagents of ~/.claude of the user as the ones of"
                        " the project, the working directory is within the home and outside of"
                        " a git repository"
                    )

                session: ClaudeCodeSession = await _session(
                    sessions,
                    identity=identity,
                    thread=thread,
                    workspace=workspace,
                )
                streamed: _Streamed = _Streamed()
                # every run gets a token which is valid now
                credential: ClaudeCodeResolvedCredential | Missing = (
                    Missing
                    if runner.credential is Missing
                    else await runner.resolving(runner.credential, binary=runner.binary)
                )
                stream = runner.running(
                    ClaudeCodeLaunch.of(
                        resolved,
                        binary=runner.binary,
                        prompt=prompt_blocks(message.content),
                        workspace=workspace,
                        instructions=text,
                        session=session,
                        credential=credential,
                        tools_server=tools_server,
                        base_url=None if runner.base_url is Missing else runner.base_url,
                        # inside of the scope of the agent, so the run continues its trace
                        trace_context=ctx.trace_context(),
                        resource_attributes={
                            "draive.agent": identity.name,
                            "draive.agent.uri": identity.uri,
                            "draive.thread": str(thread.identifier),
                        },
                    )
                )
                try:
                    async for event in stream:
                        if isinstance(event, ClaudeCodeStarted):
                            _verify_authentication(event, credential)
                            _report_started(
                                resolved,
                                event,
                                resolved.connectors,
                                tools_server=tools_server is not None,
                            )

                        chunks, streamed = _step(event, streamed)
                        for chunk in chunks:
                            yield chunk

                finally:
                    await stream.aclose()

                result: ClaudeCodeResult = _concluded(streamed)
                match result.outcome:
                    case ClaudeCodeFailure() as failure:
                        raise ClaudeCodeErrored.of(failure, session_id=result.session_id)

                    case ClaudeCodeSuccess(text=answer):
                        if not streamed.text and answer:
                            yield TextContent.of(answer)  # nothing else carried the text

                        yield _summary(result)

    return Agent(identity, executing)


@asynccontextmanager
async def _serving(
    toolbox: Toolbox,
    /,
) -> AsyncGenerator[Mapping[str, BasicValue] | None]:
    if not toolbox.tools:
        yield None
        return

    async with ClaudeCodeToolsBridge.serve(toolbox) as bridge:
        yield bridge.mcp_server


def _current_thread() -> AgentThread:
    if not ctx.contains_state(AgentThread):
        raise ClaudeCodeException("AgentThread not specified")

    return ctx.state(AgentThread)


async def _resolved(
    instructions: Template | str,
    /,
) -> str:
    if isinstance(instructions, Template):
        return await TemplatesRepository.resolve_str(instructions)

    return instructions


async def _session(
    sessions: ClaudeCodeSessions | None,
    /,
    *,
    identity: AgentIdentity,
    thread: AgentThread,
    workspace: ClaudeCodeWorkspace,
) -> ClaudeCodeSession:
    if sessions is None:
        return ClaudeCodeNoSession()

    identifier: str = ClaudeCodeSessions.identifier(identity.uri, thread.identifier)
    # also a session which did not finish is stored, and can not be started again
    if await workspace.has_session(identifier):
        return ClaudeCodeResumedSession(identifier=identifier)

    return ClaudeCodeNewSession(identifier=identifier)


def _step(
    event: ClaudeCodeEvent,
    streamed: _Streamed,
    /,
) -> tuple[Sequence[_Chunk], _Streamed]:
    match event:
        case ClaudeCodeStarted(session_id=session_id):
            return (), streamed.updating(started=session_id)

        case ClaudeCodeDelta(text=text):
            chunks, streamed = _streaming(text, streamed)
            return chunks, streamed.updating(previewed=streamed.previewed + text)

        case ClaudeCodeText(text=text, parent=parent):
            if parent is None:  # the main agent
                return _completed(text, streamed)

            return _subagent_text(text, parent), streamed

        case ClaudeCodeToolUse() | ClaudeCodeToolResult():
            return (_tool_event(event),), streamed

        case ClaudeCodeResult():
            return (), streamed.updating(result=event)


def _completed(
    text: str,
    streamed: _Streamed,
    /,
) -> tuple[Sequence[_Chunk], _Streamed]:
    chunks: Sequence[_Chunk]
    if streamed.previewed.startswith(text):  # its deltas were streamed already
        chunks = ()
        streamed = streamed.updating(previewed=streamed.previewed[len(text) :])

    else:  # a preview which the block does not continue is stale, it would never match again
        chunks, streamed = _streaming(text, streamed.updating(previewed=""))

    return chunks, streamed.updating(separated=streamed.separated or bool(text.strip()))


def _streaming(
    text: str,
    streamed: _Streamed,
    /,
) -> tuple[Sequence[_Chunk], _Streamed]:
    # only line breaks are dropped between blocks, the indentation of a block is its content
    content: str = text.lstrip(_LINE_BREAKS) if streamed.separated or not streamed.text else text
    if not content:
        return (), streamed  # blank between blocks, the paragraph separates them already

    piece: str = f"{_PARAGRAPH}{content}" if streamed.separated else content
    return (TextContent.of(piece),), streamed.updating(text=streamed.text + piece, separated=False)


def _subagent_text(
    text: str,
    parent: str,
    /,
) -> Sequence[_Chunk]:
    if not text.strip():
        return ()

    return (ProcessingEvent.of("claude_code.subagent_text", text, meta={"parent": parent}),)


def _tool_event(
    event: ClaudeCodeToolUse | ClaudeCodeToolResult,
    /,
) -> ProcessingEvent:
    match event:
        case ClaudeCodeToolUse():
            return _tool_use(event)

        case ClaudeCodeToolResult():
            return _tool_result(event)


def _tool_use(
    event: ClaudeCodeToolUse,
    /,
) -> ProcessingEvent:
    return ProcessingEvent.of(
        "claude_code.tool_use",
        meta={
            "tool": event.tool,
            "identifier": event.identifier,
            "arguments": json.dumps(event.arguments, default=str),
            **({"parent": event.parent} if event.parent else {}),
        },
    )


def _tool_result(
    event: ClaudeCodeToolResult,
    /,
) -> ProcessingEvent:
    return ProcessingEvent.of(
        "claude_code.tool_result",
        event.content,
        meta={
            "identifier": event.identifier,
            "failed": event.failed,
            **({"parent": event.parent} if event.parent else {}),
        },
    )


def _concluded(
    streamed: _Streamed,
    /,
) -> ClaudeCodeResult:
    if streamed.result is None:  # a custom runner which ended its stream without a result
        raise ClaudeCodeUnfinished(session_id=streamed.started)

    _record_result(streamed.result)
    return streamed.result


def _verify_authentication(
    started: ClaudeCodeStarted,
    credential: ClaudeCodeResolvedCredential | Missing,
    /,
) -> None:
    # never run on a local session of Claude Code - the one of the user or of an earlier login
    if credential is Missing:
        return

    if started.api_key_source != credential.reported_source:
        raise ClaudeCodeUnavailable(
            f"Claude Code authenticates with '{started.api_key_source}'"
            f" instead of the provided {credential.label}"
        )


def _connectors_of(
    started: ClaudeCodeStarted,
    connectors: ClaudeCodeConnectors,
    /,
) -> tuple[Sequence[str], Sequence[str]]:
    match connectors:
        case ClaudeCodeNoConnectors():
            return (), ()

        case ClaudeCodeAccountConnectors():
            # connectors removed by the filters are not reported by Claude Code at all
            return started.unavailable_connectors, started.failed_connectors


def _named(
    names: Sequence[str],
    /,
) -> str:
    more: int = len(names) - _REPORTED_CONNECTORS
    return ", ".join(names[:_REPORTED_CONNECTORS]) + (f" and {more} more" if more > 0 else "")


def _report_started(
    config: ClaudeCodeConfig,
    started: ClaudeCodeStarted,
    connectors: ClaudeCodeConnectors,
    /,
    *,
    tools_server: bool,
) -> None:
    unavailable: Sequence[str]
    failed: Sequence[str]
    unavailable, failed = _connectors_of(started, connectors)
    ctx.record_info(
        attributes={
            "claude_code.session": started.session_id,
            "claude_code.model": started.model,
            "claude_code.version": started.version,
            "claude_code.permission_mode": started.permission_mode,
            "claude_code.tools": list(started.tools),
            "claude_code.mcp_servers": [server.name for server in started.mcp_servers],
            "claude_code.mcp_failed": list(started.failed_mcp_servers),
            "claude_code.connectors_unavailable": list(unavailable[:_REPORTED_CONNECTORS]),
        },
    )
    if tools_server and TOOLS_SERVER_KEY in started.failed_mcp_servers:
        raise ClaudeCodeUnavailable(
            "the draive tools server of the run did not connect, the agent would run without"
            " its tools"
        )

    if started.failed_mcp_servers:
        ctx.log_warning(
            f"Claude Code could not connect MCP servers: {', '.join(started.failed_mcp_servers)}"
        )

    # connectors which need an authorization are only recorded - most were never authorized by
    # the account, a warning about them would come with every run
    if failed:
        ctx.log_warning(f"Claude Code could not connect connectors: {_named(failed)}")

    if leaked := ClaudeCodeLaunch.leaked_plugins(config, started.plugins):
        ctx.log_warning(
            "Claude Code loaded built-in plugins unknown to the isolating settings, which are"
            f" not disabled: {', '.join(leaked)}"
        )

    if enforced := ClaudeCodeLaunch.enforced_plugins(config, started.plugins):
        ctx.log_info(
            "Claude Code loaded built-in plugins enforced by the organization of the account:"
            f" {', '.join(enforced)}"
        )


def _record_result(
    result: ClaudeCodeResult,
    /,
) -> None:
    for usage in result.usage:
        record_usage_metrics(
            provider=_PROVIDER,
            model=usage.model,
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cache_read_tokens,
            output_tokens=usage.output_tokens,
        )

    match result.outcome:
        case ClaudeCodeFailure(terminal_reason=terminal_reason):
            reason: str = terminal_reason or "failed"

        case ClaudeCodeSuccess():
            reason = "completed"

    if result.cost_usd is not None:
        ctx.record_info(
            metric="agent.claude_code.cost",
            value=result.cost_usd,
            unit="usd",
            kind="counter",
            attributes={"claude_code.terminal_reason": reason},
        )

    ctx.record_info(
        attributes={
            "claude_code.session": result.session_id,
            "claude_code.turns": result.turns,
            "claude_code.failed": result.failed,
        }
    )


def _summary(
    result: ClaudeCodeResult,
    /,
) -> ProcessingEvent:
    return ProcessingEvent.of(
        "claude_code.result",
        meta={
            "session": result.session_id,
            "turns": result.turns,
            "duration_ms": result.duration_ms,
            "cost_usd": result.cost_usd,
            "permission_denials": result.permission_denials,
        },
    )
