import asyncio
import shutil
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import final
from uuid import NAMESPACE_URL, UUID, uuid5

from haiway import Missing, State

from draive.claude.config import ClaudeCodeConfig

__all__ = (
    "ClaudeCodeWorkspace",
    "claude_workspace",
    "session_exists",
    "session_identifier",
)


@final
class ClaudeCodeWorkspace(State):
    """Directories of a Claude Code run.

    Attributes
    ----------
    directory
        Working directory of the process.
    home
        Private home of the process.
    config_directory
        Configuration directory of Claude Code, it holds the sessions.
    temporary_directory
        Directory of temporary files.
    root
        Directory holding the private ones.
    """

    directory: str
    home: str
    config_directory: str
    temporary_directory: str
    root: str


def _prepared(
    root: Path,
    directory: str | Missing,
    /,
) -> ClaudeCodeWorkspace:
    home: Path = root / "home"
    config_directory: Path = root / "claude"
    temporary: Path = root / "tmp"
    for path in (home, config_directory, temporary):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)

    working: Path
    if directory is Missing:
        working = root / "workspace"
        working.mkdir(mode=0o700, parents=True, exist_ok=True)

    else:
        working = Path(directory)
        if not working.is_absolute() or not working.is_dir():
            raise ValueError(f"Claude Code directory '{directory}' has to be an existing directory")

    return ClaudeCodeWorkspace(
        directory=str(working),
        home=str(home),
        config_directory=str(config_directory),
        temporary_directory=str(temporary),
        root=str(root),
    )


@asynccontextmanager
async def claude_workspace(
    config: ClaudeCodeConfig,
    /,
) -> AsyncGenerator[ClaudeCodeWorkspace]:
    """Prepare the directories of a run.

    Ephemeral runs get private temporary directories removed after the run, resumable ones
    live within the state directory of the configuration and are kept.

    Parameters
    ----------
    config
        Configuration of the run.

    Yields
    ------
    ClaudeCodeWorkspace
        Prepared directories.

    Raises
    ------
    ValueError
        When the configured directories are not usable.
    """
    if config.persistence == "resumable":
        assert config.state_directory is not Missing  # nosec: B101
        yield await asyncio.to_thread(
            _prepared,
            Path(config.state_directory).resolve(),  # the process changes its working directory
            config.directory,
        )
        return

    root: str = await asyncio.to_thread(tempfile.mkdtemp, prefix="draive-claude-")
    try:
        yield await asyncio.to_thread(_prepared, Path(root), config.directory)

    finally:
        await asyncio.shield(asyncio.to_thread(shutil.rmtree, root, ignore_errors=True))


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
