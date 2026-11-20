import asyncio
import shutil
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import final

from haiway import Missing, State

from draive.claude.config import ClaudeCodeConfig
from draive.claude.sessions import ClaudeCodeSessions

__all__ = (
    "ClaudeCodeWorkspace",
    "claude_workspace",
)


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


@asynccontextmanager
async def claude_workspace(
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
        yield await asyncio.to_thread(
            _prepared,
            Path(sessions.state_directory).resolve(),  # the process changes its working directory
            config.directory,
        )
        return

    root: str = await asyncio.to_thread(tempfile.mkdtemp, prefix="draive-claude-")
    try:
        yield await asyncio.to_thread(_prepared, Path(root), config.directory)

    finally:
        await asyncio.shield(asyncio.to_thread(shutil.rmtree, root, ignore_errors=True))
