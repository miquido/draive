import asyncio
from pathlib import Path
from typing import Annotated, final
from uuid import NAMESPACE_URL, UUID, uuid5

from haiway import State, Verifier

__all__ = (
    "ClaudeCodeNewSession",
    "ClaudeCodeNoSession",
    "ClaudeCodeResumedSession",
    "ClaudeCodeSession",
    "ClaudeCodeSessions",
    "session_exists",
    "session_identifier",
)


def _uuid(
    value: str,
) -> str:
    UUID(value)  # Claude Code accepts nothing else as a session identifier
    return value


def _directory(
    value: str,
) -> str:
    if not value.strip():
        raise ValueError("A state directory can not be empty")

    return value


@final
class ClaudeCodeNoSession(State):
    """The run leaves no session behind."""


@final
class ClaudeCodeNewSession(State):
    """The run starts the session of the given identifier.

    Attributes
    ----------
    identifier
        UUID of the session, Claude Code refuses to start a session which exists.
    """

    identifier: Annotated[str, Verifier(_uuid)]


@final
class ClaudeCodeResumedSession(State):
    """The run continues the stored session of the given identifier.

    Attributes
    ----------
    identifier
        UUID of the stored session.
    """

    identifier: Annotated[str, Verifier(_uuid)]


type ClaudeCodeSession = ClaudeCodeNoSession | ClaudeCodeNewSession | ClaudeCodeResumedSession


@final
class ClaudeCodeSessions(State):
    """Memory of an agent kept by Claude Code, as the sessions of the threads of the agent.

    A thread continues the session it has with the whole context Claude Code had - tool use,
    compaction and prompt caching included. It is the alternative of ``AgentMemory``, which hands
    the history to a fresh session of every run.

    Attributes
    ----------
    state_directory
        Directory holding the sessions, their home and by default the working directory.
    """

    state_directory: Annotated[str, Verifier(_directory)]


def session_identifier(
    agent_uri: str,
    thread: UUID,
    /,
) -> str:
    """Identifier of the Claude Code session of an agent thread.

    Sessions of different agents never share an identifier, even within the same thread,
    sessions of the same agent and thread always do.

    Parameters
    ----------
    agent_uri
        URI of the agent.
    thread
        Identifier of the thread.

    Returns
    -------
    str
        Session identifier, a UUID as Claude Code requires.
    """
    return str(uuid5(NAMESPACE_URL, f"{agent_uri}#{thread}"))


def _transcript_exists(
    config_directory: str,
    session_id: str,
    /,
) -> bool:
    # sessions are stored by Claude Code as `projects/<project>/<session id>.jsonl`
    return any(Path(config_directory, "projects").glob(f"*/{session_id}.jsonl"))


async def session_exists(
    config_directory: str,
    session_id: str,
    /,
) -> bool:
    """Check if Claude Code stored a session, whether it finished or not.

    The transcript of a session is written while it runs, so also a run which was interrupted
    or failed leaves a session which has to be resumed - it can not be started again.

    Parameters
    ----------
    config_directory
        Configuration directory of the runs.
    session_id
        Identifier of the session.

    Returns
    -------
    bool
        Whether the session has a stored transcript.
    """
    return await asyncio.to_thread(_transcript_exists, config_directory, session_id)
