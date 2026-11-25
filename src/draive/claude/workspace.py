import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import final

from haiway import Missing, State

from draive.claude._completion import completed
from draive.claude.config import ClaudeCodeConfig
from draive.claude.sessions import ClaudeCodeSessions

__all__ = ("ClaudeCodeWorkspace",)


@final
class ClaudeCodeWorkspace(State):
    """Directories of a Claude Code run.

    Attributes
    ----------
    root
        Directory holding the private ones.
    directory
        Working directory of the process.
    """

    @classmethod
    @asynccontextmanager
    async def prepare(
        cls,
        config: ClaudeCodeConfig,
        sessions: ClaudeCodeSessions | None,
        /,
    ) -> AsyncGenerator[ClaudeCodeWorkspace]:
        """Prepare the directories of a run.

        Runs without sessions get private temporary directories removed after the run, runs with
        them live within the state directory of the sessions and are kept.

        Parameters
        ----------
        config
            Configuration of the run.
        sessions
            Sessions kept by Claude Code, when the run is a part of them.

        Yields
        ------
        ClaudeCodeWorkspace
            Prepared directories.

        Raises
        ------
        ValueError
            When the configured working directory does not exist.
        """
        if sessions is not None:
            # resolved, as the process changes its working directory
            yield await asyncio.to_thread(
                _prepared,
                Path(sessions.state_directory).resolve(),
                config.directory,
            )
            return

        root: str = tempfile.mkdtemp(prefix="draive-claude-")  # owned from here on
        try:
            yield await asyncio.to_thread(_prepared, Path(root), config.directory)

        finally:
            await completed(
                asyncio.create_task(asyncio.to_thread(shutil.rmtree, root, ignore_errors=True))
            )

    root: str
    directory: str

    @property
    def home(self) -> str:
        """Private home of the process."""
        return str(Path(self.root) / "home")

    @property
    def config_directory(self) -> str:
        """Configuration directory of Claude Code, it holds the sessions."""
        return str(Path(self.root) / "claude")

    @property
    def temporary_directory(self) -> str:
        """Directory of temporary files."""
        return str(Path(self.root) / "tmp")

    async def has_session(
        self,
        session_id: str,
        /,
    ) -> bool:
        """Check if Claude Code stored a session, whether it finished or not.

        The transcript of a session is written while it runs, so also a run which was interrupted
        or failed leaves a session which has to be resumed - it can not be started again.

        Parameters
        ----------
        session_id
            Identifier of the session.

        Returns
        -------
        bool
            Whether the session has a stored transcript.
        """
        return await asyncio.to_thread(_transcript_exists, self.config_directory, session_id)

    async def reaches_user_agents(self) -> bool:
        """Check if project settings of the run would load the subagents of the user.

        Claude Code loads the subagents of the ``.claude`` directory of every parent of the
        working directory up to its git repository, which reaches ``~/.claude/agents`` of the
        user when the working directory is within the home and outside of a repository below it.

        Returns
        -------
        bool
            Whether the user has subagents which the run would load.
        """
        return await asyncio.to_thread(_reaches_user_agents, self.directory)


def _reaches_user_agents(
    directory: str,
    /,
) -> bool:
    home: Path = Path(os.path.realpath(os.path.expanduser("~")))
    if not (home / ".claude" / "agents").is_dir():
        return False

    working: Path = Path(os.path.realpath(directory))
    if not working.is_relative_to(home):
        return False

    # a repository below the home ends the walk before it
    below_home: list[Path] = [
        parent
        for parent in (working, *working.parents)
        if parent != home and parent.is_relative_to(home)
    ]
    return not any((parent / ".git").exists() for parent in below_home)


def _transcript_exists(
    config_directory: str,
    session_id: str,
    /,
) -> bool:
    # sessions are stored by Claude Code as `projects/<project>/<session id>.jsonl`
    return any(Path(config_directory, "projects").glob(f"*/{session_id}.jsonl"))


def _prepared(
    root: Path,
    directory: str | Missing,
    /,
) -> ClaudeCodeWorkspace:
    working: Path
    if directory is Missing:
        working = root / "workspace"

    else:
        working = Path(directory)
        if not working.is_dir():
            raise ValueError(f"Claude Code directory '{directory}' has to be an existing directory")

    workspace: ClaudeCodeWorkspace = ClaudeCodeWorkspace(root=str(root), directory=str(working))
    for path in (workspace.home, workspace.config_directory, workspace.temporary_directory):
        Path(path).mkdir(mode=0o700, parents=True, exist_ok=True)

    working.mkdir(mode=0o700, parents=True, exist_ok=True)
    return workspace
