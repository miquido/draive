from typing import Annotated, Final, final

from haiway import Sensitive, State, Verifier

__all__ = (
    "MAX_CREDENTIAL_BYTES",
    "ClaudeCodeApiKey",
    "ClaudeCodeCredential",
    "ClaudeCodeOAuthToken",
)

# a credential is written to a pipe before the process which reads it exists, it has to fit the
# capacity of a pipe, which POSIX requires to be larger than that
MAX_CREDENTIAL_BYTES: Final[int] = 4096


def _credential(
    value: str,
) -> str:
    if not value or len(value.encode()) > MAX_CREDENTIAL_BYTES:
        raise ValueError(f"A credential has to have between 1 and {MAX_CREDENTIAL_BYTES} bytes")

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
        return "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"

    @property
    def reported_source(self) -> str:
        """Authentication source Claude Code reports when this token is the only credential."""
        return "none"  # a token is not reported as an API key

    @property
    def label(self) -> str:
        return "OAuth token"


type ClaudeCodeCredential = ClaudeCodeApiKey | ClaudeCodeOAuthToken
