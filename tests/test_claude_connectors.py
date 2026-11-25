from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from haiway import Missing, ctx

from draive.claude import (
    ClaudeCodeAccessToken,
    ClaudeCodeAccountConnectors,
    ClaudeCodeApiKey,
    ClaudeCodeConfig,
    ClaudeCodeNoConnectors,
    ClaudeCodeOAuthToken,
    ClaudeCodeUnavailable,
)
from draive.claude.launch import (
    build_launch,
    claude_arguments,
    claude_environment,
    verify_connectors,
)
from draive.claude.workspace import ClaudeCodeWorkspace

WORKSPACE = ClaudeCodeWorkspace(root="/private", directory="/private/work")
TOKEN = "sk-ant-oat01-access-token-value"


def _access_token(*scopes: str) -> ClaudeCodeAccessToken:
    return ClaudeCodeAccessToken(
        value=TOKEN,
        scopes=scopes,
        expires_at=datetime.now(UTC) + timedelta(hours=2),
    )


def _environment(
    config: ClaudeCodeConfig,
    credential: Any = Missing,
) -> Any:
    return claude_environment(config, workspace=WORKSPACE, parent={}, credential=credential)


def test_connectors_are_off_by_default() -> None:
    assert ClaudeCodeConfig().connectors == ClaudeCodeNoConnectors()
    assert "--strict-mcp-config" in claude_arguments(ClaudeCodeConfig())
    assert "MCP_CONNECTION_NONBLOCKING" not in _environment(ClaudeCodeConfig())


@pytest.mark.parametrize(
    ("connectors", "waiting"),
    [(ClaudeCodeAccountConnectors(), True), (ClaudeCodeAccountConnectors(wait=False), False)],
)
def test_account_connectors_are_not_strict(
    connectors: ClaudeCodeAccountConnectors,
    waiting: bool,
) -> None:
    config = ClaudeCodeConfig(connectors=connectors)

    assert "--strict-mcp-config" not in claude_arguments(config)
    environment = _environment(config)
    assert environment.get("MCP_CONNECTION_NONBLOCKING") == ("0" if waiting else None)


def test_account_connectors_keep_the_configured_servers() -> None:
    config = ClaudeCodeConfig(
        connectors=ClaudeCodeAccountConnectors(),
        mcp_servers={"docs": {"type": "http", "url": "https://docs.example"}},
    )

    arguments = claude_arguments(config)

    assert "--strict-mcp-config" not in arguments
    assert "docs" in arguments[arguments.index("--mcp-config") + 1]


def test_access_tokens_declare_their_scopes_and_never_leak() -> None:
    token = _access_token("user:inference", "user:mcp_servers")
    config = ClaudeCodeConfig(connectors=ClaudeCodeAccountConnectors())

    launch = build_launch(
        config,
        binary="/bin/claude",
        prompt=[{"type": "text", "text": "hello"}],
        workspace=WORKSPACE,
        credential=token,
    )

    assert launch.environment["CLAUDE_CODE_OAUTH_SCOPES"] == "user:inference user:mcp_servers"
    assert launch.credential == token
    assert TOKEN not in " ".join(launch.arguments)
    assert TOKEN not in " ".join(launch.environment.values())
    assert TOKEN not in repr(token) + repr(launch)
    assert token.reported_source == "none"
    assert token.descriptor == "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"
    assert TOKEN in launch.redactions


@pytest.mark.parametrize(
    "credential",
    [
        ClaudeCodeApiKey(value="sk-ant-api03-key"),
        ClaudeCodeOAuthToken(value="sk-ant-oat01-token"),
        _access_token("user:inference"),
        Missing,
    ],
)
def test_account_connectors_refuse_credentials_without_the_scope(credential: Any) -> None:
    with pytest.raises(ClaudeCodeUnavailable):
        verify_connectors(ClaudeCodeConfig(connectors=ClaudeCodeAccountConnectors()), credential)


def test_connectors_are_accepted_with_the_scope() -> None:
    verify_connectors(
        ClaudeCodeConfig(connectors=ClaudeCodeAccountConnectors()),
        _access_token("user:inference", "user:mcp_servers"),
    )


@pytest.mark.parametrize(
    "credential",
    [ClaudeCodeApiKey(value="sk-ant-api03-key"), _access_token("user:inference"), Missing],
)
def test_no_connectors_accept_any_credential(credential: Any) -> None:
    verify_connectors(ClaudeCodeConfig(), credential)


@pytest.mark.asyncio
async def test_connectors_are_a_part_of_the_scoped_configuration() -> None:
    async with ctx.scope("test", ClaudeCodeConfig(connectors=ClaudeCodeAccountConnectors())):
        assert ctx.state(ClaudeCodeConfig).connectors == ClaudeCodeAccountConnectors()


def test_launches_refuse_connectors_with_credentials_without_the_scope() -> None:
    with pytest.raises(ClaudeCodeUnavailable):
        build_launch(
            ClaudeCodeConfig(connectors=ClaudeCodeAccountConnectors()),
            binary="/bin/claude",
            prompt=[],
            workspace=WORKSPACE,
            credential=ClaudeCodeApiKey(value="sk-ant-api03-key"),
        )
