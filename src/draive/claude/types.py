from collections.abc import AsyncGenerator
from typing import Annotated, Literal, Protocol, final, runtime_checkable

from haiway import Missing, Sensitive, State

from draive.agents.types import AgentException
from draive.claude.events import ClaudeCodeEvent
from draive.claude.launch import ClaudeCodeLaunch

__all__ = (
    "ClaudeCodeException",
    "ClaudeCodeFailed",
    "ClaudeCodeRunner",
    "ClaudeCodeRunning",
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
        super().__init__(f"Claude Code is unavailable: {reason}")
        self.reason: str = reason


@final
class ClaudeCodeFailed(ClaudeCodeException):
    """Raised when a Claude Code run failed.

    Parameters
    ----------
    reason
        Description of the failure, i.e. the error reported by the run.
    session_id
        Claude Code session of the run, when it got to report one.
    api_error_status
        HTTP status of the API failure which ended the run, when there was one.
    terminal_reason
        Reason reported by Claude Code for ending the run, i.e. ``"api_error"``.
    exit_code
        Exit code of the process, when it was available.
    stderr
        Tail of the process error output with credentials redacted.
    """

    __slots__ = (
        "api_error_status",
        "exit_code",
        "reason",
        "session_id",
        "stderr",
        "terminal_reason",
    )

    def __init__(
        self,
        reason: str,
        *,
        session_id: str | None = None,
        api_error_status: int | None = None,
        terminal_reason: str | None = None,
        exit_code: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(f"Claude Code run failed: {reason}")
        self.reason: str = reason
        self.session_id: str | None = session_id
        self.api_error_status: int | None = api_error_status
        self.terminal_reason: str | None = terminal_reason
        self.exit_code: int | None = exit_code
        self.stderr: str = stderr


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
        API key or OAuth token, never part of the representation of the state.
    credential_kind
        Kind of the credential.
    base_url
        Custom API endpoint, i.e. a gateway.
    """

    binary: str
    running: ClaudeCodeRunning
    credential: Annotated[str, Sensitive()] | Missing = Missing
    credential_kind: Literal["api_key", "oauth_token"] = "api_key"
    base_url: str | Missing = Missing
