import json
from collections.abc import AsyncGenerator, Sequence
from typing import Final, final
from urllib.parse import quote

from haiway import Meta, MetaValues, Missing, State, ctx

from draive.agents import Agent, AgentIdentity, AgentMemory, AgentMessage, AgentThread
from draive.claude.config import ClaudeCodeConfig
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
from draive.claude.launch import build_launch
from draive.claude.login import ClaudeCodeLogins
from draive.claude.sessions import (
    ClaudeCodeNewSession,
    ClaudeCodeNoSession,
    ClaudeCodeResumedSession,
    ClaudeCodeSession,
    ClaudeCodeSessions,
    session_exists,
    session_identifier,
)
from draive.claude.transcript import prompt_blocks, render_transcript
from draive.claude.types import (
    ClaudeCodeEnded,
    ClaudeCodeErrored,
    ClaudeCodeException,
    ClaudeCodeRunner,
    ClaudeCodeUnavailable,
)
from draive.claude.workspace import ClaudeCodeWorkspace, claude_workspace
from draive.models import ModelContext, ModelInput, ModelOutput, record_usage_metrics
from draive.multimodal import (
    MultimodalContent,
    MultimodalContentPart,
    Template,
    TemplatesRepository,
    TextContent,
)
from draive.utils import ProcessingEvent

__all__ = ("claude_code_agent",)

_PROVIDER: Final[str] = "claude_code"
# built-in plugins which an organization enforces for its accounts, no setting disables them
_ENFORCED_PLUGINS: Final[Sequence[str]] = ("cc-plugin-sec-default",)

type _Chunk = MultimodalContentPart | ProcessingEvent


@final
class _Continuity(State):
    """How a run continues the conversation, the only source of its session and its history."""

    session: ClaudeCodeSession
    recalled: ModelContext = ()


@final
class _Streamed(State):
    """What was streamed of a run so far, the only source of its answer and its result."""

    previewed: str = ""  # streamed text which its complete block is not to repeat
    answer: Sequence[str] = ()
    result: ClaudeCodeResult | None = None

    def echoes(
        self,
        text: str,
        /,
    ) -> bool:
        """Whether the complete block only repeats what was streamed of it already."""
        return self.previewed.startswith(text)


def claude_code_agent(
    agent: AgentIdentity | str,
    *,
    description: str = "",
    instructions: Template | str = "",
    memory: AgentMemory | ClaudeCodeSessions = AgentMemory.disabled,
    config: ClaudeCodeConfig | None = None,
    meta: Meta | MetaValues | None = None,
) -> Agent:
    """Create an agent which runs Claude Code, fully managed.

    Every call runs the ``claude`` executable as a process isolated from the user setup of
    Claude Code, streams its work and stops it whatever way the call ends. The text of the
    agent is streamed as text parts, its tool use and results as ``ProcessingEvent``s named
    ``claude_code.tool_use``, ``claude_code.tool_result`` and ``claude_code.result``.

    The memory of the conversation is either managed by draive or by Claude Code. With
    ``AgentMemory`` the recalled history is handed to a fresh session and the turn remembered
    afterwards, which works with any memory backend. With ``ClaudeCodeSessions`` Claude Code
    keeps the session of each thread and resumes it, with the whole context it had. Nothing is
    kept by default.

    The call is an async generator, close it when it is left early, i.e. with ``ctx.closing``.

    Parameters
    ----------
    agent
        Identity of the agent, or its name. The identity made of a name is stable, so agents
        of the same name share the sessions and the memory of their threads, also after a
        restart.
    description
        Description of the agent, ignored when an identity is given.
    instructions
        Instructions of the agent, appended to the system prompt of Claude Code, or replacing
        it according to the configuration.
    memory
        Memory of the conversation, disabled by default.
    config
        Configuration of the runs, ``ClaudeCodeConfig`` of the scope when not set.
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
        When the message or the working directory is not supported.
    """
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
            async with claude_workspace(
                resolved,
                memory if isinstance(memory, ClaudeCodeSessions) else None,
            ) as workspace:
                continuity: _Continuity = await _continuity(
                    memory,
                    identity=identity,
                    thread=thread,
                    workspace=workspace,
                    instructions=text,
                )
                streamed: _Streamed = _Streamed()
                # every run gets a token which is valid now
                credential: ClaudeCodeResolvedCredential | Missing = (
                    Missing
                    if runner.credential is Missing
                    else await (
                        ClaudeCodeLogins() if runner.logins is Missing else runner.logins
                    ).resolve(runner.credential, binary=runner.binary)
                )
                stream = runner.running(
                    build_launch(
                        resolved,
                        binary=runner.binary,
                        prompt=prompt_blocks(
                            message.content,
                            history=render_transcript(continuity.recalled)
                            if continuity.recalled
                            else None,
                        ),
                        workspace=workspace,
                        instructions=text,
                        session=continuity.session,
                        credential=credential,
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
                            _report_started(event)

                        for chunk in _chunks(event, streamed):
                            yield chunk

                        streamed = _accumulated(event, streamed)

                finally:
                    await stream.aclose()

                result: ClaudeCodeResult = _concluded(streamed)
                match result.outcome:
                    case ClaudeCodeFailure() as failure:
                        raise ClaudeCodeErrored(failure=failure, session_id=result.session_id)

                    case ClaudeCodeSuccess(text=answer):
                        if not streamed.answer and answer:
                            yield TextContent.of(answer)  # nothing else carried the text

                        if isinstance(memory, AgentMemory):
                            await memory.remember(
                                thread=thread,
                                context=(
                                    *continuity.recalled,
                                    ModelInput.of(message.content, meta=message.meta),
                                    ModelOutput.of(
                                        MultimodalContent.of("".join(streamed.answer) or answer)
                                    ),
                                ),
                            )

                        yield _summary(result)

    return Agent(identity, executing)


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


async def _continuity(
    memory: AgentMemory | ClaudeCodeSessions,
    /,
    *,
    identity: AgentIdentity,
    thread: AgentThread,
    workspace: ClaudeCodeWorkspace,
    instructions: str,
) -> _Continuity:
    match memory:
        case ClaudeCodeSessions():
            identifier: str = session_identifier(identity.uri, thread.identifier)
            # also a session which did not finish is stored, and can not be started again
            if await session_exists(workspace.config_directory, identifier):
                return _Continuity(session=ClaudeCodeResumedSession(identifier=identifier))

            return _Continuity(session=ClaudeCodeNewSession(identifier=identifier))

        case AgentMemory():
            await memory.prepare(thread=thread, instructions=instructions)
            return _Continuity(
                session=ClaudeCodeNoSession(),
                recalled=await memory.recall(thread=thread, context=()),
            )


def _chunks(
    event: ClaudeCodeEvent,
    streamed: _Streamed,
    /,
) -> Sequence[_Chunk]:
    match event:
        case ClaudeCodeDelta(text=text):
            return (TextContent.of(text),)

        case ClaudeCodeText(text=text, parent=str() as parent):
            return (ProcessingEvent.of("claude_code.subagent_text", text, meta={"parent": parent}),)

        case ClaudeCodeText(text=text):
            return () if streamed.echoes(text) else (TextContent.of(text),)

        case ClaudeCodeToolUse():
            return (_tool_use(event),)

        case ClaudeCodeToolResult():
            return (_tool_result(event),)

        case _:
            return ()


def _accumulated(
    event: ClaudeCodeEvent,
    streamed: _Streamed,
    /,
) -> _Streamed:
    match event:
        case ClaudeCodeDelta(text=text):
            return streamed.updating(
                previewed=streamed.previewed + text,
                answer=(*streamed.answer, text),
            )

        case ClaudeCodeText(text=text, parent=None):
            if streamed.echoes(text):
                return streamed.updating(previewed=streamed.previewed[len(text) :])

            return streamed.updating(answer=(*streamed.answer, text))

        case ClaudeCodeResult():
            return streamed.updating(result=event)

        case _:
            return streamed


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
    if streamed.result is None:  # a runner without a result would be a broken one
        raise ClaudeCodeEnded()

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


def _report_started(
    started: ClaudeCodeStarted,
    /,
) -> None:
    ctx.record_info(
        attributes={
            "claude_code.session": started.session_id,
            "claude_code.model": started.model,
            "claude_code.version": started.version,
            "claude_code.permission_mode": started.permission_mode,
            "claude_code.tools": list(started.tools),
            "claude_code.mcp_servers": [server.name for server in started.mcp_servers],
            "claude_code.mcp_failed": list(started.failed_mcp_servers),
        },
    )
    if started.failed_mcp_servers:
        ctx.log_warning(
            f"Claude Code could not connect MCP servers: {', '.join(started.failed_mcp_servers)}"
        )

    if leaked := [
        name
        for name in started.plugins
        if name.startswith("cc-plugin-") and name not in _ENFORCED_PLUGINS
    ]:
        ctx.log_warning(
            f"Claude Code loaded built-in plugins which are not disabled: {', '.join(leaked)}"
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
