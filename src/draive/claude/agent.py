import json
from collections.abc import AsyncGenerator, Sequence
from typing import Final
from urllib.parse import quote

from haiway import Meta, MetaValues, Missing, ctx

from draive.agents import Agent, AgentIdentity, AgentMemory, AgentMessage, AgentThread
from draive.claude.config import ClaudeCodeConfig
from draive.claude.events import (
    ClaudeCodeDelta,
    ClaudeCodeResult,
    ClaudeCodeStarted,
    ClaudeCodeText,
    ClaudeCodeToolResult,
    ClaudeCodeToolUse,
)
from draive.claude.launch import ClaudeCodeSession, build_launch, validate_config
from draive.claude.transcript import prompt_blocks, render_transcript
from draive.claude.types import (
    ClaudeCodeException,
    ClaudeCodeFailed,
    ClaudeCodeRunner,
    ClaudeCodeUnavailable,
)
from draive.claude.workspace import claude_workspace, session_exists, session_identifier
from draive.models import (
    ModelContext,
    ModelInput,
    ModelOutput,
    record_usage_metrics,
)
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


def claude_code_agent(  # noqa: C901, PLR0915
    agent: AgentIdentity | str,
    *,
    description: str = "",
    instructions: Template | str = "",
    memory: AgentMemory = AgentMemory.disabled,
    config: ClaudeCodeConfig | None = None,
    meta: Meta | MetaValues | None = None,
) -> Agent:
    """Create an agent which runs Claude Code, fully managed.

    Every call runs the ``claude`` executable as a process isolated from the user setup of
    Claude Code, streams its work and stops it whatever way the call ends. The text of the
    agent is streamed as text parts, its tool use and results as ``ProcessingEvent``s named
    ``claude_code.tool_use``, ``claude_code.tool_result`` and ``claude_code.result``.

    Memory of the conversation is either managed by draive or by Claude Code. With
    ``memory`` the recalled history is handed to a fresh session and the turn remembered
    afterwards, which works with any memory backend. With the ``resumable`` persistence of
    the configuration Claude Code keeps the sessions of threads and resumes them - the memory
    is not used then. Nothing is kept by default.

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
        Memory of the conversation managed by draive, disabled by default.
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
        When the configuration or the message is not supported.
    """
    identity: AgentIdentity = (
        # stable, as the sessions of resumable runs are found by it after a restart
        AgentIdentity.of(
            f"agent://claude-code/{quote(agent, safe='')}",
            name=agent,
            description=description,
            meta=meta,
        )
        if isinstance(agent, str)
        else agent
    )

    async def executing(  # noqa: C901, PLR0912, PLR0915
        message: AgentMessage,
    ) -> AsyncGenerator[MultimodalContentPart | ProcessingEvent]:
        async with ctx.scope("agent.claude_code"):
            if not ctx.contains_state(AgentThread):
                raise ClaudeCodeException("AgentThread not specified")

            thread: AgentThread = ctx.state(AgentThread)
            resolved: ClaudeCodeConfig = config or ctx.state(ClaudeCodeConfig)
            validate_config(resolved)
            runner: ClaudeCodeRunner = ctx.state(ClaudeCodeRunner)
            resumable: bool = resolved.persistence == "resumable"
            if resumable and memory is not AgentMemory.disabled:
                raise ValueError(
                    "Resumable Claude Code sessions are the memory of the agent,"
                    " use either them or AgentMemory"
                )

            resolved_instructions: str = (
                await TemplatesRepository.resolve_str(instructions)
                if isinstance(instructions, Template)
                else instructions
            )
            recalled: ModelContext = ()
            if not resumable:
                await memory.prepare(thread=thread, instructions=resolved_instructions)
                recalled = await memory.recall(thread=thread, context=())

            blocks = prompt_blocks(
                message.content,
                history=render_transcript(recalled) if recalled else None,
            )
            async with claude_workspace(resolved) as workspace:
                session: ClaudeCodeSession = ClaudeCodeSession()
                if resumable:
                    # also a session which did not finish is stored, and can not be started again
                    identifier: str = session_identifier(identity.uri, thread.identifier)
                    session = ClaudeCodeSession(
                        mode=(
                            "resume"
                            if await session_exists(workspace.config_directory, identifier)
                            else "new"
                        ),
                        identifier=identifier,
                    )

                launch = build_launch(
                    resolved,
                    binary=runner.binary,
                    prompt=blocks,
                    directory=workspace.directory,
                    home=workspace.home,
                    config_directory=workspace.config_directory,
                    temporary_directory=workspace.temporary_directory,
                    instructions=resolved_instructions,
                    session=session,
                    credential=_present(runner.credential),
                    credential_kind=runner.credential_kind,
                    base_url=_present(runner.base_url),
                    # inside of the scope of the agent, so the run continues its trace
                    trace_context=ctx.trace_context(),
                    resource_attributes={
                        "draive.agent": identity.name,
                        "draive.agent.uri": identity.uri,
                        "draive.thread": str(thread.identifier),
                    },
                )

                answer: list[str] = []
                previewed: str = ""  # streamed text the complete block is not repeated for
                result: ClaudeCodeResult | None = None
                stream = runner.running(launch)
                try:
                    async for event in stream:
                        if isinstance(event, ClaudeCodeStarted):
                            _verify_authentication(event, runner)
                            ctx.record_info(
                                attributes={
                                    "claude_code.session": event.session_id,
                                    "claude_code.model": event.model,
                                    "claude_code.version": event.version,
                                    "claude_code.permission_mode": event.permission_mode,
                                    "claude_code.tools": list(event.tools),
                                    "claude_code.mcp_servers": list(event.mcp_servers),
                                    "claude_code.mcp_failed": list(event.mcp_failed),
                                }
                            )
                            if event.mcp_failed:
                                ctx.log_warning(
                                    "Claude Code could not connect MCP servers:"
                                    f" {', '.join(event.mcp_failed)}"
                                )

                            if leaked := [
                                name
                                for name in event.plugins
                                if name.startswith("cc-plugin-") and name not in _ENFORCED_PLUGINS
                            ]:
                                ctx.log_warning(
                                    "Claude Code loaded built-in plugins which are not disabled:"
                                    f" {', '.join(leaked)}"
                                )

                        elif isinstance(event, ClaudeCodeDelta):
                            previewed += event.text
                            answer.append(event.text)
                            yield TextContent.of(event.text)

                        elif isinstance(event, ClaudeCodeText):
                            if event.parent is not None:
                                yield ProcessingEvent.of(
                                    "claude_code.subagent_text",
                                    event.text,
                                    meta={"parent": event.parent},
                                )

                            elif previewed.startswith(event.text):
                                previewed = previewed[len(event.text) :]  # already streamed

                            else:
                                answer.append(event.text)
                                yield TextContent.of(event.text)

                        elif isinstance(event, ClaudeCodeToolUse):
                            yield ProcessingEvent.of(
                                "claude_code.tool_use",
                                meta={
                                    "tool": event.tool,
                                    "identifier": event.identifier,
                                    "arguments": json.dumps(event.arguments, default=str),
                                    **({"parent": event.parent} if event.parent else {}),
                                },
                            )

                        elif isinstance(event, ClaudeCodeToolResult):
                            yield ProcessingEvent.of(
                                "claude_code.tool_result",
                                event.content,
                                meta={
                                    "identifier": event.identifier,
                                    "failed": event.failed,
                                    **({"parent": event.parent} if event.parent else {}),
                                },
                            )

                        else:
                            result = event

                finally:
                    await stream.aclose()

                if result is None:  # a runner without a result would be a broken one
                    raise ClaudeCodeFailed("the run ended without a result")

                _record_result(result)
                if result.failed:
                    raise ClaudeCodeFailed(
                        result.text or result.subtype or "the run failed",
                        session_id=result.session_id,
                        api_error_status=result.api_error_status,
                        terminal_reason=result.terminal_reason,
                    )

                if not answer and result.text:
                    answer.append(result.text)  # nothing was streamed, the result is the text
                    yield TextContent.of(result.text)

                final: str = "".join(answer)
                if not resumable:
                    await memory.remember(
                        thread=thread,
                        context=(
                            *recalled,
                            ModelInput.of(message.content, meta=message.meta),
                            ModelOutput.of(MultimodalContent.of(final)),
                        ),
                    )

                yield ProcessingEvent.of(
                    "claude_code.result",
                    meta={
                        "session": result.session_id,
                        "turns": result.turns,
                        "duration_ms": result.duration_ms,
                        "cost_usd": result.cost_usd,
                        "permission_denials": result.permission_denials,
                    },
                )

    return Agent(identity, executing)


def _verify_authentication(
    started: ClaudeCodeStarted,
    runner: ClaudeCodeRunner,
    /,
) -> None:
    # never run on a local session of Claude Code - the one of the user or of an earlier login
    if runner.credential is Missing:
        return

    # an API key is reported by the variable it is taken from, an OAuth token is not an API key
    # at all - Claude Code reports no key source for it
    expected: str
    label: str
    if runner.credential_kind == "api_key":
        expected, label = "ANTHROPIC_API_KEY", "API key"

    else:
        expected, label = "none", "OAuth token"

    if started.api_key_source != expected:
        raise ClaudeCodeUnavailable(
            f"Claude Code authenticates with '{started.api_key_source}'"
            f" instead of the provided {label}"
        )


def _present(
    value: str | Missing,
    /,
) -> str | None:
    return None if value is Missing else value


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

    if result.cost_usd is not None:
        ctx.record_info(
            metric="agent.claude_code.cost",
            value=result.cost_usd,
            unit="usd",
            kind="counter",
            attributes={"claude_code.terminal_reason": result.terminal_reason or "completed"},
        )

    ctx.record_info(
        attributes={
            "claude_code.session": result.session_id,
            "claude_code.turns": result.turns,
            "claude_code.failed": result.failed,
        }
    )
