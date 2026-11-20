import json
from typing import Any

import pytest

from draive.claude import (
    ClaudeCodeDelta,
    ClaudeCodeResult,
    ClaudeCodeStarted,
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
                    {"name": "broken", "status": "failed"},
                    {"name": "unknown"},
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
        mcp_servers=("docs", "broken", "unknown"),
        mcp_failed=("broken",),
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
    assert (event.failed, event.text, event.turns, event.duration_ms) == (False, "done", 3, 1200)
    assert event.cost_usd == 0.25
    assert event.permission_denials == 1
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
    assert (event.failed, event.api_error_status) == (failed, status)


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
