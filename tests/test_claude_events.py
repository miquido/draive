import json
from typing import Any

import pytest

from draive.claude import (
    ClaudeCodeDelta,
    ClaudeCodeFailure,
    ClaudeCodeMCPServer,
    ClaudeCodeResult,
    ClaudeCodeStarted,
    ClaudeCodeSuccess,
    ClaudeCodeText,
    ClaudeCodeToolResult,
    ClaudeCodeToolUse,
)
from draive.claude.events import parse_events


def _line(payload: Any) -> str:
    return json.dumps(payload)


def test_init_is_parsed_into_started() -> None:
    (event,) = parse_events(
        _line(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s1",
                "model": "claude-sonnet-5-5",
                "claude_code_version": "2.1.329",
                "permissionMode": "dontAsk",
                "apiKeySource": "ANTHROPIC_API_KEY",
                "tools": ["Read", "Grep"],
                "mcp_servers": [
                    {"name": "docs", "status": "connected"},
                    {"name": "broken", "status": "failed", "source": "claudeai"},
                    {"name": "unknown"},
                    {"name": "claude.ai Slack", "status": "needs-auth", "source": "claudeai"},
                ],
                "plugins": [{"name": "cc-plugin-privacy", "path": "builtin"}],
            }
        )
    )

    assert event == ClaudeCodeStarted(
        session_id="s1",
        model="claude-sonnet-5-5",
        version="2.1.329",
        permission_mode="dontAsk",
        api_key_source="ANTHROPIC_API_KEY",
        tools=("Read", "Grep"),
        mcp_servers=(
            ClaudeCodeMCPServer(name="docs", status="connected"),
            ClaudeCodeMCPServer(name="broken", status="failed", source="claudeai"),
            ClaudeCodeMCPServer(name="unknown"),  # no status is reported by every version
            ClaudeCodeMCPServer(name="claude.ai Slack", status="needs-auth", source="claudeai"),
        ),
        plugins=("cc-plugin-privacy",),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "system", "subtype": "status", "status": "requesting", "session_id": "s1"},
        {"type": "system", "subtype": "init"},  # no session
        {"type": "rate_limit_event"},
        {"type": "control_request"},
        {"no": "type"},
    ],
)
def test_other_lines_are_ignored(payload: Any) -> None:
    assert parse_events(_line(payload)) == ()


@pytest.mark.parametrize("line", ["", "not json", "[1, 2]", '"text"', "123", "null", b"\xff\xfe"])
def test_lines_which_are_not_objects_are_ignored(line: Any) -> None:
    assert parse_events(line) == ()


def test_assistant_message_carries_text_and_tool_use_in_order() -> None:
    events = parse_events(
        _line(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "reading"},
                        {"type": "text", "text": "   "},
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a"}},
                        {"type": "tool_use", "id": 5, "name": "Bad"},
                    ]
                },
            }
        )
    )

    assert events == (
        ClaudeCodeText(text="reading"),
        ClaudeCodeToolUse(identifier="t1", tool="Read", arguments={"path": "a"}),
    )


def test_subagent_messages_keep_their_parent() -> None:
    events = parse_events(
        _line(
            {
                "type": "assistant",
                "parent_tool_use_id": "agent-1",
                "message": {"content": [{"type": "text", "text": "sub"}]},
            }
        )
    )

    assert events == (ClaudeCodeText(text="sub", parent="agent-1"),)


def test_synthetic_error_messages_are_not_text() -> None:
    assert (
        parse_events(
            _line(
                {
                    "type": "assistant",
                    "error": "authentication_failed",
                    "message": {"content": [{"type": "text", "text": "Invalid API key"}]},
                }
            )
        )
        == ()
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain", "plain"),
        ([{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}], "a\nb"),
        (None, ""),
    ],
)
def test_tool_results_flatten_their_content(content: Any, expected: str) -> None:
    (event,) = parse_events(
        _line(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": content,
                            "is_error": True,
                        },
                        {"type": "text", "text": "ignored"},
                    ]
                },
            }
        )
    )

    assert event == ClaudeCodeToolResult(identifier="t1", content=expected, failed=True)


def test_text_deltas_of_the_main_agent_are_previews() -> None:
    delta = {
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hel"}},
    }

    assert parse_events(_line(delta)) == (ClaudeCodeDelta(text="hel"),)
    assert parse_events(_line({**delta, "parent_tool_use_id": "agent-1"})) == ()
    assert (
        parse_events(
            _line(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "input_json_delta", "partial_json": "{"},
                    },
                }
            )
        )
        == ()
    )
    assert parse_events(_line({"type": "stream_event", "event": {"type": "message_start"}})) == ()


def test_result_carries_cost_and_usage() -> None:
    (event,) = parse_events(
        _line(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "session_id": "s1",
                "result": "done",
                "num_turns": 3,
                "duration_ms": 1200,
                "total_cost_usd": 0.25,
                "permission_denials": [{"tool": "Bash"}],
                "modelUsage": {
                    "claude-sonnet-5-5": {
                        "inputTokens": 10,
                        "outputTokens": 5,
                        "cacheReadInputTokens": 2,
                        "cacheCreationInputTokens": 1,
                        "costUSD": 0.25,
                    }
                },
            }
        )
    )

    assert isinstance(event, ClaudeCodeResult)
    assert event.outcome == ClaudeCodeSuccess(text="done")
    assert not event.failed
    assert (event.turns, event.duration_ms) == (3, 1200)
    (usage,) = event.usage
    assert (usage.model, usage.input_tokens, usage.output_tokens) == ("claude-sonnet-5-5", 10, 5)
    assert (usage.cache_read_tokens, usage.cache_creation_tokens, usage.cost_usd) == (2, 1, 0.25)


@pytest.mark.parametrize(
    ("fields", "failed", "status"),
    [
        ({"subtype": "success", "is_error": True, "api_error_status": 401}, True, 401),
        ({"subtype": "error_max_turns", "is_error": False}, True, None),
        ({"subtype": "error_during_execution"}, True, None),
        ({"subtype": "success", "is_error": False, "api_error_status": 0}, False, None),
        ({"is_error": False}, False, None),
    ],
)
def test_result_failure_is_not_only_its_subtype(
    fields: dict[str, Any], failed: bool, status: int | None
) -> None:
    (event,) = parse_events(_line({"type": "result", "session_id": "s", **fields}))

    assert isinstance(event, ClaudeCodeResult)
    assert event.failed is failed
    match event.outcome:
        case ClaudeCodeFailure(api_error_status=api_error_status):
            assert api_error_status == status

        case ClaudeCodeSuccess():
            assert status is None


def test_failed_result_describes_itself_by_its_text_or_subtype() -> None:
    (with_text,) = parse_events(
        _line(
            {
                "type": "result",
                "is_error": True,
                "result": "Invalid API key",
                "api_error_status": 401,
                "terminal_reason": "api_error",
            }
        )
    )
    (with_subtype,) = parse_events(_line({"type": "result", "subtype": "error_max_turns"}))
    (with_nothing,) = parse_events(_line({"type": "result", "is_error": True}))

    assert isinstance(with_text, ClaudeCodeResult)
    assert with_text.outcome == ClaudeCodeFailure(
        reason="Invalid API key", api_error_status=401, terminal_reason="api_error"
    )
    assert isinstance(with_subtype, ClaudeCodeResult)
    assert with_subtype.outcome == ClaudeCodeFailure(reason="error_max_turns")
    assert isinstance(with_nothing, ClaudeCodeResult)
    assert with_nothing.outcome == ClaudeCodeFailure(reason="the run failed")


def test_started_derives_its_failed_servers_from_their_status() -> None:
    (event,) = parse_events(
        _line(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s",
                "mcp_servers": [
                    {"name": "a", "status": "connected"},
                    {"name": "b", "status": "needs-auth"},
                    {"name": "c"},
                    # an account connector never authenticated is the choice of the account
                    {"name": "claude.ai Slack", "status": "needs-auth", "source": "claudeai"},
                    {"name": "claude.ai Drive", "status": "failed", "source": "claudeai"},
                    {"name": "claude.ai Docs", "status": "connected", "source": "claudeai"},
                ],
            }
        )
    )

    assert isinstance(event, ClaudeCodeStarted)
    assert event.failed_mcp_servers == ("b",)  # connectors are reported apart
    assert event.unavailable_connectors == ("claude.ai Slack", "claude.ai Drive")
    assert event.failed_connectors == ("claude.ai Drive",)


def test_malformed_result_fields_fall_back_to_defaults() -> None:
    (event,) = parse_events(
        _line(
            {
                "type": "result",
                "num_turns": "3",
                "duration_ms": True,
                "total_cost_usd": "free",
                "modelUsage": {"m": "oops", "n": {"inputTokens": "x"}},
                "permission_denials": "none",
            }
        )
    )

    assert isinstance(event, ClaudeCodeResult)
    assert (event.turns, event.duration_ms) == (0, 0)
    assert (event.cost_usd, event.permission_denials) == (None, 0)
    assert [(usage.model, usage.input_tokens) for usage in event.usage] == [("n", 0)]


def test_started_separates_unavailable_connectors_from_failed_servers() -> None:
    (event,) = parse_events(
        _line(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s",
                "mcp_servers": [
                    {"name": "claude.ai A", "status": "connected", "source": "claudeai"},
                    {"name": "claude.ai B", "status": "failed", "source": "claudeai"},
                    {"name": "claude.ai C", "status": "needs-auth", "source": "claudeai"},
                    {"name": "claude.ai D", "status": "pending", "source": "claudeai"},
                    {"name": "local", "status": "failed", "source": "project"},
                ],
            }
        )
    )

    assert isinstance(event, ClaudeCodeStarted)
    assert event.unavailable_connectors == ("claude.ai B", "claude.ai C")
    assert event.failed_mcp_servers == ("local",)


@pytest.mark.parametrize(
    ("fields", "authentication"),
    [
        ({"api_error_status": 401}, True),
        ({"api_error_type": "authentication_error"}, True),
        ({"api_error_status": 401, "api_error_type": "authentication_error"}, True),
        ({"api_error_status": 429, "api_error_type": "rate_limit_error"}, False),
        ({"api_error_status": 500}, False),
        ({}, False),
    ],
)
def test_failure_knows_whether_the_credential_was_rejected(
    fields: dict[str, Any],
    authentication: bool,
) -> None:
    (event,) = parse_events(
        _line({"type": "result", "subtype": "success", "is_error": True, **fields})
    )

    assert isinstance(event, ClaudeCodeResult)
    assert isinstance(event.outcome, ClaudeCodeFailure)
    assert event.outcome.authentication_failed is authentication
    assert event.outcome.api_error_type == fields.get("api_error_type")


@pytest.mark.parametrize(
    ("status", "expected", "connected"),
    [
        ("connected", "connected", True),
        ("failed", "failed", False),
        ("needs-auth", "needs-auth", False),
        ("pending", "pending", False),
        ("disabled", "disabled", False),
        ("something-new", "unknown", False),
        (5, None, True),
        (None, None, True),
    ],
)
def test_the_status_of_a_server_is_closed_and_decides_whether_it_is_connected(
    status: Any,
    expected: str | None,
    connected: bool,
) -> None:
    (event,) = parse_events(
        _line(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s",
                "mcp_servers": [{"name": "a", "status": status}],
            }
        )
    )

    assert isinstance(event, ClaudeCodeStarted)
    (server,) = event.mcp_servers
    assert server.status == expected
    assert server.connected is connected
