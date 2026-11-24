from abc import ABC, abstractmethod
from typing import final

from draive.agents.types import AgentException
from draive.claude.events import ClaudeCodeFailure

__all__ = (
    "ClaudeCodeAuthenticationFailed",
    "ClaudeCodeEnded",
    "ClaudeCodeErrored",
    "ClaudeCodeException",
    "ClaudeCodeFailed",
    "ClaudeCodeLoginUnusable",
    "ClaudeCodeOutputTooLarge",
    "ClaudeCodeTimedOut",
    "ClaudeCodeUnavailable",
)


class ClaudeCodeException(AgentException):
    """Base exception of Claude Code runs."""


class ClaudeCodeUnavailable(ClaudeCodeException):
    """Raised when Claude Code can not be started - missing binary or credentials, bad setup.

    ``ClaudeCodeLoginUnusable`` is the only variant of it.

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


@final
class ClaudeCodeLoginUnusable(ClaudeCodeUnavailable):
    """A ``ClaudeCodeLocalLogin`` gives no valid access token - log in again.

    Its credentials file is missing, unreadable or malformed, a leftover of a login kept in the
    keychain (macOS), or its token has expired and could not be refreshed. Logging in again
    (``claude /login`` as the user of the login) is what helps. Still a ``ClaudeCodeUnavailable``.

    Parameters
    ----------
    reason
        Description of why the login is unusable, never carrying its tokens.
    """

    __slots__ = ()


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


class ClaudeCodeErrored(ClaudeCodeFailed):
    """The run ended with an error reported by Claude Code.

    ``ClaudeCodeAuthenticationFailed`` is the only variant of it.

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

    @staticmethod
    def of(
        failure: ClaudeCodeFailure,
        *,
        session_id: str | None = None,
    ) -> ClaudeCodeErrored:
        """Prepare the exception of the failure - the variant when the credential was rejected.

        Parameters
        ----------
        failure
            The error the run ended with.
        session_id
            Claude Code session of the run, when it got to report one.

        Returns
        -------
        ClaudeCodeErrored
            ``ClaudeCodeAuthenticationFailed`` for an authentication failure, the base otherwise.
        """
        if failure.authentication_failed:
            return ClaudeCodeAuthenticationFailed(failure=failure, session_id=session_id)

        return ClaudeCodeErrored(failure=failure, session_id=session_id)

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
class ClaudeCodeAuthenticationFailed(ClaudeCodeErrored):
    """The API rejected the credential of the run - HTTP 401 or an authentication error.

    A token of a ``ClaudeCodeLocalLogin`` which was revoked or rotated by another process fails
    this way, refreshing the login and running again may help. Still a ``ClaudeCodeErrored``.

    Parameters
    ----------
    failure
        The error the run ended with, it never carries the credential.
    session_id
        Claude Code session of the run, when it got to report one.
    """

    __slots__ = ()


@final
class ClaudeCodeOutputTooLarge(ClaudeCodeFailed):
    """A line of the output was larger than the run can read."""

    @property
    def reason(self) -> str:
        return "the output line is too large"
