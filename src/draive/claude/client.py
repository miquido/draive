import os
import shutil
import weakref
from collections.abc import Iterable
from types import TracebackType
from typing import Final, final

from haiway import Missing, State, ValidationError, getenv_str

from draive.claude.credentials import (
    ClaudeCodeApiKey,
    ClaudeCodeCredential,
    ClaudeCodeLocalLogin,
    ClaudeCodeOAuthToken,
)
from draive.claude.login import ClaudeCodeLogins
from draive.claude.process import run_process
from draive.claude.types import ClaudeCodeRunner, ClaudeCodeUnavailable

__all__ = ("ClaudeCode",)

_API_KEY_PREFIX: Final[str] = "sk-ant-api"
_OAUTH_TOKEN_PREFIX: Final[str] = "sk-ant-oat"


@final
class ClaudeCode:
    """Access to Claude Code, prepares the capability of running it for agents.

    Claude Code is run as a managed process of the ``claude`` executable, nothing of it is
    imported - there is no additional dependency. The executable and the credential are
    verified when the scope is entered.

    Runs always authenticate with the credential, and are stopped if Claude Code reports to
    authenticate in any other way. By default they never use the login of the user - its
    session, keychain and configuration are out of reach of the isolated process. With
    ``ClaudeCodeLocalLogin`` draive reads only a short-lived access token from the credentials
    file of the login, refreshing it under a lock through Claude Code itself, and the run still
    gets a private home and configuration - never the directory of the login nor its refresh
    token. Without a credential entering the scope fails.

    Parameters
    ----------
    credential
        ``ClaudeCodeApiKey``, ``ClaudeCodeOAuthToken`` (``claude setup-token``) or
        ``ClaudeCodeLocalLogin`` - the only one which allows account connectors. Resolved from
        ``ANTHROPIC_API_KEY`` or ``CLAUDE_CODE_OAUTH_TOKEN`` when not set. A key given as the
        token or the other way is refused unless a custom endpoint is used.
    binary
        Path of the ``claude`` executable. Resolved from ``CLAUDE_CODE_BINARY`` and the
        ``PATH`` when not set.
    base_url
        Custom API endpoint, i.e. a gateway.

    Examples
    --------
    >>> async with ctx.scope(
    ...     "coding",
    ...     ClaudeCodeConfig(model="sonnet", tools=("Read", "Edit")),
    ...     disposables=(ClaudeCode(credential=ClaudeCodeApiKey("sk-ant-api...")),),
    ... ):
    ...     coder = claude_code_agent("coder", instructions="Fix the bugs.")
    ...     async for chunk in coder.call(input="Fix the failing test"):
    ...         ...

    Raises
    ------
    ValueError
        When the kind of the credential does not match its value.
    ClaudeCodeUnavailable
        When entering the scope and the executable, the credential or the credentials file of
        a local login is missing.
    """

    __slots__ = (
        "_base_url",
        "_binary",
        "_credential",
        "_logins",
    )

    def __init__(
        self,
        *,
        credential: ClaudeCodeCredential | None = None,
        binary: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if credential is not None:
            _verify_kind(credential, custom_endpoint=bool(base_url))

        self._credential: ClaudeCodeCredential | None = credential
        self._binary: str | None = binary
        self._base_url: str | None = base_url
        # one per entry, made in its context so what its refreshes report reaches its scope - exits
        # can not be told apart (haiway enters and exits from different tasks), so each exit waits
        # for the refreshes of every live entry; an entry is gone once its runner is released
        self._logins: weakref.WeakSet[ClaudeCodeLogins] = weakref.WeakSet()

    def __repr__(self) -> str:
        return "ClaudeCode()"  # nothing of the credential

    async def __aenter__(self) -> Iterable[State]:
        credential: ClaudeCodeCredential = self._credential or _environment_credential()
        _verify_kind(credential, custom_endpoint=bool(self._base_url))
        if isinstance(credential, ClaudeCodeLocalLogin):
            await ClaudeCodeLogins.verify(
                credential
            )  # fails at startup, the token is resolved per run

        binary: str = _resolved_binary(self._binary)
        logins: ClaudeCodeLogins = ClaudeCodeLogins()
        self._logins.add(logins)
        return (
            ClaudeCodeRunner(
                resolving=logins.resolve,
                binary=binary,
                running=run_process,
                credential=credential,
                base_url=self._base_url or Missing,
            ),
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # runs release their processes on their own, refreshes of a login are waited for - one
        # cut off by the end of the event loop could lose the rotated refresh token
        for logins in tuple(self._logins):
            await logins.drain()


def _environment_credential() -> ClaudeCodeCredential:
    try:
        if key := getenv_str("ANTHROPIC_API_KEY"):
            return ClaudeCodeApiKey(value=key)

        if token := getenv_str("CLAUDE_CODE_OAUTH_TOKEN"):
            return ClaudeCodeOAuthToken(value=token)

    except ValidationError as exc:
        raise ClaudeCodeUnavailable(f"the credential of the environment is invalid: {exc}") from exc

    raise ClaudeCodeUnavailable(
        "no credentials, provide a credential or set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN"
    )


def _verify_kind(
    credential: ClaudeCodeCredential,
    /,
    *,
    custom_endpoint: bool,
) -> None:
    # the kinds are told apart by their prefixes, a gateway is free to use any other credential
    if custom_endpoint:
        return

    match credential:
        case ClaudeCodeApiKey(value=value) if value.startswith(_OAUTH_TOKEN_PREFIX):
            raise ValueError("The given API key is an OAuth token, use ClaudeCodeOAuthToken")

        case ClaudeCodeOAuthToken(value=value) if value.startswith(_API_KEY_PREFIX):
            raise ValueError("The given OAuth token is an API key, use ClaudeCodeApiKey")

        case ClaudeCodeApiKey() | ClaudeCodeOAuthToken() | ClaudeCodeLocalLogin():
            pass  # a local login has no value to be told by


def _resolved_binary(
    binary: str | None,
    /,
) -> str:
    candidates: list[str | None] = [
        binary,
        getenv_str("CLAUDE_CODE_BINARY"),
        shutil.which("claude"),
    ]
    for candidate in candidates:
        if not candidate:
            continue

        resolved: str = os.path.realpath(os.path.expanduser(candidate))
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved

        if candidate is binary:  # an explicit choice is not replaced with another one
            break

    raise ClaudeCodeUnavailable(
        "the claude executable was not found, install Claude Code or set CLAUDE_CODE_BINARY"
    )
