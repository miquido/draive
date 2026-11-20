import os
import shutil
from collections.abc import Iterable
from types import TracebackType
from typing import Final, Literal, final

from haiway import Missing, State, getenv_str

from draive.claude.process import run_process
from draive.claude.types import ClaudeCodeRunner, ClaudeCodeUnavailable

__all__ = ("ClaudeCode",)

_DEFAULT_BINARY: Final[str] = "~/.local/bin/claude"
_API_KEY_PREFIX: Final[str] = "sk-ant-api"
_OAUTH_TOKEN_PREFIX: Final[str] = "sk-ant-oat"


@final
class ClaudeCode:
    """Access to Claude Code, prepares the capability of running it for agents.

    Claude Code is run as a managed process of the ``claude`` executable, nothing of it is
    imported - there is no additional dependency. The executable and the credentials are
    verified when the scope is entered.

    Runs always authenticate with the given credential. They never use the login of the
    user - its session, keychain and configuration are out of reach of the isolated process -
    and are stopped if Claude Code reports to authenticate in any other way. Without a
    credential entering the scope fails.

    Parameters
    ----------
    binary
        Path of the ``claude`` executable. Resolved from ``CLAUDE_CODE_BINARY``, the
        ``PATH`` and ``~/.local/bin/claude`` when not set.
    api_key
        Anthropic API key, ``ANTHROPIC_API_KEY`` when not set.
    oauth_token
        Claude Code OAuth token (``claude setup-token``), ``CLAUDE_CODE_OAUTH_TOKEN`` when not
        set. Use either the key or the token, a key given as the token or the other way is
        refused unless a custom endpoint is used.
    base_url
        Custom API endpoint, i.e. a gateway.

    Examples
    --------
    >>> async with ctx.scope(
    ...     "coding",
    ...     ClaudeCodeConfig(model="sonnet", tools=("Read", "Edit")),
    ...     disposables=(ClaudeCode(),),
    ... ):
    ...     coder = claude_code_agent("coder", instructions="Fix the bugs.")
    ...     async for chunk in coder.call(input="Fix the failing test"):
    ...         ...
    """

    __slots__ = (
        "_api_key",
        "_base_url",
        "_binary",
        "_oauth_token",
    )

    def __init__(
        self,
        *,
        binary: str | None = None,
        api_key: str | None = None,
        oauth_token: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if api_key and oauth_token:
            raise ValueError("Claude Code takes either an API key or an OAuth token, not both")

        _verify_kind(api_key, oauth_token, custom_endpoint=bool(base_url))

        self._binary: str | None = binary
        self._api_key: str | None = api_key
        self._oauth_token: str | None = oauth_token
        self._base_url: str | None = base_url

    def __repr__(self) -> str:
        return "ClaudeCode()"  # nothing of the credentials

    async def __aenter__(self) -> Iterable[State]:
        credential: str
        kind: Literal["api_key", "oauth_token"]
        if self._api_key:
            credential, kind = self._api_key, "api_key"

        elif self._oauth_token:
            credential, kind = self._oauth_token, "oauth_token"

        elif key := getenv_str("ANTHROPIC_API_KEY"):
            credential, kind = key, "api_key"
            _verify_kind(key, None, custom_endpoint=bool(self._base_url))

        elif token := getenv_str("CLAUDE_CODE_OAUTH_TOKEN"):
            credential, kind = token, "oauth_token"
            _verify_kind(None, token, custom_endpoint=bool(self._base_url))

        else:
            raise ClaudeCodeUnavailable(
                "no credentials, provide an API key or an OAuth token or set"
                " ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN"
            )

        return (
            ClaudeCodeRunner(
                binary=_resolved_binary(self._binary),
                running=run_process,
                credential=credential,
                credential_kind=kind,
                base_url=self._base_url or Missing,
            ),
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass  # runs release their processes on their own


def _verify_kind(
    api_key: str | None,
    oauth_token: str | None,
    /,
    *,
    custom_endpoint: bool,
) -> None:
    # the kinds are told apart by their prefixes, a gateway is free to use any other credential
    if custom_endpoint:
        return

    if api_key and api_key.startswith(_OAUTH_TOKEN_PREFIX):
        raise ValueError("The given API key is an OAuth token, pass it as oauth_token")

    if oauth_token and oauth_token.startswith(_API_KEY_PREFIX):
        raise ValueError("The given OAuth token is an API key, pass it as api_key")


def _resolved_binary(
    binary: str | None,
    /,
) -> str:
    candidates: list[str | None] = [
        binary,
        getenv_str("CLAUDE_CODE_BINARY"),
        shutil.which("claude"),
        os.path.expanduser(_DEFAULT_BINARY),
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
