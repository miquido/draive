from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Protocol, final, runtime_checkable

from haiway import Missing, State

from draive.agents.types import AgentException
from draive.claude.credentials import ClaudeCodeCredential
from draive.claude.events import ClaudeCodeEvent, ClaudeCodeFailure
from draive.claude.launch import ClaudeCodeLaunch

__all__ = (
    "ClaudeCodeEnded",
    "ClaudeCodeErrored",
    "ClaudeCodeException",
    "ClaudeCodeFailed",
    "ClaudeCodeOutputTooLarge",
    "ClaudeCodeRunner",
    "ClaudeCodeRunning",
    "ClaudeCodeTimedOut",
    "ClaudeCodeUnavailable",
)


class ClaudeCodeException(AgentException):
    """Base exception of Claude Code runs."""


@final
class ClaudeCodeUnavailable(ClaudeCodeException):
    """Raised when Claude Code can not be started - missing binary or credentials, bad setup.

    Parameters
    ----------
    reason
        Description of what is missing or invalid, never carrying credentials.
    """

    __slots__ = ("reason",)

    def __init__(
        self,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.reason: str = reason

    def __str__(self) -> str:
        return f"Claude Code is unavailable: {self.reason}"


class ClaudeCodeFailed(ClaudeCodeException, ABC):
    """Raised when a Claude Code run failed, one of the variants below.

    Parameters
    ----------
    session_id
        Claude Code session of the run, when it got to report one.
    """

    __slots__ = ("session_id",)

    def __init__(
        self,
        *,
        session_id: str | None = None,
    ) -> None:
        super().__init__()
        self.session_id: str | None = session_id

    @property
    @abstractmethod
    def reason(self) -> str:
        """Description of the failure."""

    def __str__(self) -> str:
        return f"Claude Code run failed: {self.reason}"


@final
class ClaudeCodeTimedOut(ClaudeCodeFailed):
    """The run did not end before its deadline.

    Parameters
    ----------
    timeout
        Limit of the run duration which passed, in seconds.
    session_id
        Claude Code session of the run, when it got to report one.
    """

    __slots__ = ("timeout",)

    def __init__(
        self,
        *,
        timeout: float,
        session_id: str | None = None,
    ) -> None:
        super().__init__(session_id=session_id)
        self.timeout: float = timeout

    @property
    def reason(self) -> str:
        return f"timed out after {self.timeout:g} seconds"


@final
class ClaudeCodeEnded(ClaudeCodeFailed):
    """The process ended without reporting a result.

    Parameters
    ----------
    exit_code
        Exit code of the process, when it was available.
    stderr
        Tail of the process error output with credentials redacted.
    session_id
        Claude Code session of the run, when it got to report one.
    """

    __slots__ = ("exit_code", "stderr")

    def __init__(
        self,
        *,
        exit_code: int | None = None,
        stderr: str = "",
        session_id: str | None = None,
    ) -> None:
        super().__init__(session_id=session_id)
        self.exit_code: int | None = exit_code
        self.stderr: str = stderr

    @property
    def reason(self) -> str:
        return "the process ended without a result"


@final
class ClaudeCodeErrored(ClaudeCodeFailed):
    """The run ended with an error reported by Claude Code.

    Parameters
    ----------
    failure
        The error the run ended with.
    session_id
        Claude Code session of the run, when it got to report one.
    """

    __slots__ = ("failure",)

    def __init__(
        self,
        *,
        failure: ClaudeCodeFailure,
        session_id: str | None = None,
    ) -> None:
        super().__init__(session_id=session_id)
        self.failure: ClaudeCodeFailure = failure

    @property
    def reason(self) -> str:
        return self.failure.reason

    @property
    def api_error_status(self) -> int | None:
        """HTTP status of the API failure which ended the run, when it was one."""
        return self.failure.api_error_status

    @property
    def terminal_reason(self) -> str | None:
        """Reason reported by Claude Code for ending the run, i.e. ``api_error``."""
        return self.failure.terminal_reason


@final
class ClaudeCodeOutputTooLarge(ClaudeCodeFailed):
    """A line of the output was larger than the run can read."""

    @property
    def reason(self) -> str:
        return "the output line is too large"


@runtime_checkable
class ClaudeCodeRunning(Protocol):
    """Protocol of the run of a prepared Claude Code launch, streaming its events.

    Implementations release the process within their own ``try``/``finally``, so ending,
    closing or finalizing the stream stops the run.
    """

    def __call__(
        self,
        launch: ClaudeCodeLaunch,
        /,
    ) -> AsyncGenerator[ClaudeCodeEvent]: ...


@final
class ClaudeCodeRunner(State):
    """Installed capability of running Claude Code, prepared by ``ClaudeCode``.

    Attributes
    ----------
    binary
        Absolute path of the Claude Code executable.
    running
        Implementation of the run, replace it to fake the process in tests.
    credential
        The only credential of the runs, never part of the representation of the state.
    base_url
        Custom API endpoint, i.e. a gateway.
    """

    binary: str
    running: ClaudeCodeRunning
    credential: ClaudeCodeCredential | Missing = Missing
    base_url: str | Missing = Missing
