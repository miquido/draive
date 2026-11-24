from collections.abc import AsyncGenerator
from typing import Protocol, final, runtime_checkable

from haiway import Missing, State

from draive.claude.credentials import ClaudeCodeCredential
from draive.claude.events import ClaudeCodeEvent
from draive.claude.exceptions import (
    ClaudeCodeEnded,
    ClaudeCodeErrored,
    ClaudeCodeException,
    ClaudeCodeFailed,
    ClaudeCodeOutputTooLarge,
    ClaudeCodeTimedOut,
    ClaudeCodeUnavailable,
)
from draive.claude.launch import ClaudeCodeLaunch
from draive.claude.login import ClaudeCodeLogins

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
    logins
        Owner of the refreshes of local logins, prepared by ``ClaudeCode``.
    base_url
        Custom API endpoint, i.e. a gateway.
    """

    binary: str
    running: ClaudeCodeRunning
    credential: ClaudeCodeCredential | Missing = Missing
    logins: ClaudeCodeLogins | Missing = Missing
    base_url: str | Missing = Missing
