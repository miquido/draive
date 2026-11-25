from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Final, final

from haiway import Sensitive, State, Verifier

__all__ = (
    "DEFAULT_LOGIN_DIRECTORY",
    "MAX_CREDENTIAL_BYTES",
    "ClaudeCodeAccessToken",
    "ClaudeCodeApiKey",
    "ClaudeCodeCredential",
    "ClaudeCodeLocalLogin",
    "ClaudeCodeOAuthToken",
    "ClaudeCodeResolvedCredential",
)

# a credential is written to a pipe before the process which reads it exists, it has to fit the
# capacity of a pipe, which POSIX requires to be larger than that
MAX_CREDENTIAL_BYTES: Final[int] = 4096

# default configuration directory of Claude Code, where it keeps its login
DEFAULT_LOGIN_DIRECTORY: Final[str] = "~/.claude"

# both kinds of OAuth credential are handed over the same way and reported the same way
_OAUTH_DESCRIPTOR: Final[str] = "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"
_OAUTH_REPORTED_SOURCE: Final[str] = "none"  # a token is not reported as an API key


def _credential(
    value: str,
) -> str:
    if not value or len(value.encode()) > MAX_CREDENTIAL_BYTES:
        raise ValueError(f"A credential has to have between 1 and {MAX_CREDENTIAL_BYTES} bytes")

    return value


def _aware(
    value: datetime,
) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A moment has to carry its timezone")

    return value


def _not_negative(
    value: float,
) -> float:
    if value < 0:
        raise ValueError("Value can not be negative")

    return value


@final
class ClaudeCodeApiKey(State):
    """Anthropic API key, the credential of runs which are billed to an API account.

    Attributes
    ----------
    value
        Value of the key, never a part of the representation of the state.
    """

    value: Annotated[str, Sensitive(), Verifier(_credential)]

    @property
    def descriptor(self) -> str:
        """Variable pointing Claude Code to the file descriptor holding the key."""
        return "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"

    @property
    def reported_source(self) -> str:
        """Authentication source Claude Code reports when this key is the only credential."""
        return "ANTHROPIC_API_KEY"

    @property
    def label(self) -> str:
        return "API key"


@final
class ClaudeCodeOAuthToken(State):
    """Claude Code OAuth token, the credential of runs which use a subscription.

    Attributes
    ----------
    value
        Value of the token, never a part of the representation of the state.
    """

    value: Annotated[str, Sensitive(), Verifier(_credential)]

    @property
    def descriptor(self) -> str:
        """Variable pointing Claude Code to the file descriptor holding the token."""
        return _OAUTH_DESCRIPTOR

    @property
    def reported_source(self) -> str:
        """Authentication source Claude Code reports when this token is the only credential."""
        return _OAUTH_REPORTED_SOURCE

    @property
    def label(self) -> str:
        return "OAuth token"


@final
class ClaudeCodeAccessToken(State):
    """Short-lived OAuth access token of a Claude Code login, resolved from a
    ``ClaudeCodeLocalLogin`` for a single run.

    Attributes
    ----------
    value
        Value of the token, never a part of the representation of the state.
    scopes
        Scopes the token was granted, i.e. ``user:inference`` and ``user:mcp_servers``.
    expires_at
        Moment the token expires, timezone-aware, a run which outlives it fails.
    """

    value: Annotated[str, Sensitive(), Verifier(_credential)]
    scopes: Sequence[str]
    expires_at: Annotated[datetime, Verifier(_aware)]

    @property
    def descriptor(self) -> str:
        """Variable pointing Claude Code to the file descriptor holding the token."""
        return _OAUTH_DESCRIPTOR

    @property
    def reported_source(self) -> str:
        """Authentication source Claude Code reports when this token is the only credential."""
        return _OAUTH_REPORTED_SOURCE

    @property
    def label(self) -> str:
        return "access token of a login"


@final
class ClaudeCodeLocalLogin(State):
    """Credential of a local Claude Code login - a claude.ai account logged in with
    ``claude /login`` on this machine, as the user the process runs as.

    It is a source of credentials, not a value: every run reads a short-lived access token from
    the credentials file of the login (``<config_directory>/.credentials.json`` - logins kept in
    the keychain, as on macOS, are not supported). A token which is about to expire
    is refreshed first through Claude Code itself (``claude auth status``), within a lock so
    only one draive process refreshes at a time - the lock does not coordinate with an
    interactive Claude Code of the user, which may refresh the same login. Whether ``claude auth
    status`` refreshes at a margin as large as the default is not verified, Claude Code may do so
    only near the expiry: a refresh which does not extend the token is logged and the token is
    used while it is valid, without another attempt for a minute. Only the access token reaches
    the run, never the refresh token nor the directory of the login. It is the only credential
    with the ``user:mcp_servers`` scope the connectors of the account need, ``claude setup-token``
    tokens are inference only.

    A run holds an access token and can not refresh it, a run which outlives its token fails.
    Refreshing a login may also revoke the access tokens other processes still hold. Keep
    ``refresh_margin`` longer than your longest run.

    Attributes
    ----------
    config_directory
        Configuration directory of the login, ``CLAUDE_CONFIG_DIR`` of Claude Code.
    refresh_margin
        Remaining lifetime in seconds below which the token is refreshed before a run.
    """

    config_directory: str = DEFAULT_LOGIN_DIRECTORY
    refresh_margin: Annotated[float, Verifier(_not_negative)] = 1800.0


type ClaudeCodeCredential = ClaudeCodeApiKey | ClaudeCodeOAuthToken | ClaudeCodeLocalLogin
type ClaudeCodeResolvedCredential = ClaudeCodeApiKey | ClaudeCodeOAuthToken | ClaudeCodeAccessToken
