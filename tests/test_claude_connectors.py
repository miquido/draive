import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from haiway import Missing, ValidationError, ctx

from draive.claude import (
    ClaudeCodeAccessToken,
    ClaudeCodeAccountConnectors,
    ClaudeCodeAllConnectors,
    ClaudeCodeApiKey,
    ClaudeCodeConfig,
    ClaudeCodeConnectorsAt,
    ClaudeCodeNoConnectors,
    ClaudeCodeOAuthToken,
    ClaudeCodeUnavailable,
)
from draive.claude.launch import (
    build_launch,
    claude_arguments,
    claude_environment,
    claude_settings,
    verify_allowed_servers,
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


def _settings(config: ClaudeCodeConfig) -> Any:
    return json.loads(claude_settings(config))


def test_connectors_have_no_filters_by_default() -> None:
    settings = _settings(ClaudeCodeConfig(connectors=ClaudeCodeAccountConnectors()))

    assert "deniedMcpServers" not in settings
    assert "allowedMcpServers" not in settings


def test_excluded_connectors_are_denied_by_their_display_names_and_urls() -> None:
    config = ClaudeCodeConfig(
        connectors=ClaudeCodeAccountConnectors(
            excluded=("Slack", "Claude Data"),
            excluded_urls=("https://mcp.atlassian.com/*",),
        )
    )

    settings = _settings(config)

    assert settings["deniedMcpServers"] == [
        {"serverName": "claude.ai Slack"},
        {"serverName": "claude.ai Claude Data"},
        {"serverUrl": "https://mcp.atlassian.com/*"},
    ]
    assert "allowedMcpServers" not in settings


def test_allowed_connectors_are_matched_by_urls_and_the_configured_servers_are_kept() -> None:
    config = ClaudeCodeConfig(
        connectors=ClaudeCodeAccountConnectors(
            allowed=ClaudeCodeConnectorsAt(urls=("https://mcp.slack.com/*",))
        ),
        mcp_servers={
            "docs": {"type": "http", "url": "https://docs.example"},
            "local": {"command": "local-server"},
        },
    )

    settings = _settings(config)

    assert settings["allowedMcpServers"] == [
        {"serverUrl": "https://mcp.slack.com/*"},
        {"serverName": "docs"},
        {"serverName": "local"},
    ]


def test_configured_filters_are_added_to_the_derived_ones() -> None:
    config = ClaudeCodeConfig(
        connectors=ClaudeCodeAccountConnectors(
            excluded=("Slack",),
            allowed=ClaudeCodeConnectorsAt(urls=("https://mcp.slack.com/*",)),
        ),
        settings={
            "deniedMcpServers": [{"serverName": "claude.ai Slack"}, {"serverName": "other"}],
            "allowedMcpServers": [{"serverName": "extra"}],
            "custom": 1,
        },
    )

    settings = _settings(config)

    assert settings["deniedMcpServers"] == [
        {"serverName": "claude.ai Slack"},
        {"serverName": "other"},
    ]
    assert settings["allowedMcpServers"] == [
        {"serverUrl": "https://mcp.slack.com/*"},
        {"serverName": "extra"},
    ]
    assert settings["custom"] == 1


def test_configured_filters_are_left_as_they_are_without_connectors_filters() -> None:
    config = ClaudeCodeConfig(
        connectors=ClaudeCodeAccountConnectors(),
        settings={"allowedMcpServers": [{"serverName": "x"}]},
    )

    assert _settings(config)["allowedMcpServers"] == [{"serverName": "x"}]


def test_filters_are_not_sent_without_connectors() -> None:
    settings = _settings(ClaudeCodeConfig(mcp_servers={"docs": {"command": "docs"}}))

    assert "deniedMcpServers" not in settings
    assert "allowedMcpServers" not in settings


@pytest.mark.parametrize("key", ["claudeMdExcludes", "allowedMcpServers", "deniedMcpServers"])
@pytest.mark.parametrize("value", ["claude.ai Slack", {"serverName": "x"}, 1, None])
def test_configured_filters_which_are_not_lists_can_not_be_created(key: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        ClaudeCodeConfig(settings={key: value})


def test_configured_filters_which_are_lists_are_accepted() -> None:
    ClaudeCodeConfig(settings={"deniedMcpServers": [], "allowedMcpServers": ({"serverName": "a"},)})


def test_excluded_connectors_are_named_by_their_display_names() -> None:
    with pytest.raises(ValidationError):
        ClaudeCodeAccountConnectors(excluded=("claude.ai Slack",))


def test_allowed_connectors_need_urls() -> None:
    with pytest.raises(ValidationError):
        ClaudeCodeConnectorsAt(urls=())


def test_every_connector_is_allowed_by_default() -> None:
    assert ClaudeCodeAccountConnectors().allowed == ClaudeCodeAllConnectors()


@pytest.mark.parametrize("name", ["my server", "docs.v2", "dócs", "a/b"])
def test_servers_which_an_allowlist_can_not_name_are_refused_with_allowed_connectors(
    name: str,
) -> None:
    config = ClaudeCodeConfig(
        connectors=ClaudeCodeAccountConnectors(
            allowed=ClaudeCodeConnectorsAt(urls=("https://mcp.slack.com/*",))
        ),
        mcp_servers={name: {"command": "x"}},
    )

    with pytest.raises(ClaudeCodeUnavailable, match="can not be allowed"):
        verify_allowed_servers(config)

    with pytest.raises(ClaudeCodeUnavailable, match="can not be allowed"):
        build_launch(
            config,
            binary="/bin/claude",
            prompt=[],
            workspace=WORKSPACE,
            credential=_access_token("user:inference", "user:mcp_servers"),
        )


def test_any_server_name_is_fine_without_an_allowlist() -> None:
    verify_allowed_servers(
        ClaudeCodeConfig(
            connectors=ClaudeCodeAccountConnectors(),
            mcp_servers={"my server": {"command": "x"}},
        )
    )
    verify_allowed_servers(ClaudeCodeConfig(mcp_servers={"my server": {"command": "x"}}))


def test_connector_filters_survive_serialization() -> None:
    config = ClaudeCodeConfig(
        connectors=ClaudeCodeAccountConnectors(
            excluded=("Slack",),
            excluded_urls=("https://a/*",),
            allowed=ClaudeCodeConnectorsAt(urls=("https://b/*",)),
        )
    )

    assert ClaudeCodeConfig.from_mapping(config.to_mapping()) == config
