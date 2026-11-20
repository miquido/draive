import json
from pathlib import Path
from typing import Any

import pytest
from haiway import Missing

from draive.claude import ClaudeCodeConfig, ClaudeCodeTelemetry
from draive.claude.launch import (
    ClaudeCodeSession,
    build_launch,
    claude_arguments,
    claude_environment,
    claude_settings,
    validate_config,
)


def _value(arguments: Any, flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


def test_default_arguments_isolate_and_deny_by_default() -> None:
    arguments = claude_arguments(ClaudeCodeConfig())

    assert arguments[:7] == (
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--setting-sources",
    )
    assert _value(arguments, "--setting-sources") == ""  # an explicit empty argument
    assert "--strict-mcp-config" in arguments
    assert "--no-chrome" in arguments
    assert _value(arguments, "--permission-prompts") == "none"
    assert _value(arguments, "--permission-mode") == "dontAsk"
    assert _value(arguments, "--tools") == "Read,Grep,Glob"
    # file tools are allowed within the directories of the run only
    assert _value(arguments, "--allowedTools") == "Read(./**),Grep(./**),Glob(./**)"
    assert "--disable-slash-commands" in arguments
    assert "--no-session-persistence" in arguments
    for absent in (
        "--system-prompt",
        "--append-system-prompt",
        "--max-turns",
        "--max-budget-usd",
        "--effort",
        "--include-partial-messages",
        "--mcp-config",
        "--agents",
        "--add-dir",
        "--disallowedTools",
        "--plugin-dir",
    ):
        assert absent not in arguments


def test_setting_sources_are_empty_unless_configured() -> None:
    assert _value(claude_arguments(ClaudeCodeConfig()), "--setting-sources") == ""
    assert (
        _value(
            claude_arguments(ClaudeCodeConfig(setting_sources=("project", "local"))),
            "--setting-sources",
        )
        == "project,local"
    )


def test_file_tools_are_scoped_to_the_directories_of_the_run_by_default() -> None:
    arguments = claude_arguments(
        ClaudeCodeConfig(
            tools=("Read", "Edit", "Bash", "WebFetch"),
            additional_directories=("/data/a", "/data/b"),
        )
    )

    assert _value(arguments, "--allowedTools").split(",") == [
        "Read(./**)",
        "Read(//data/a/**)",
        "Read(//data/b/**)",
        "Edit(./**)",
        "Edit(//data/a/**)",
        "Edit(//data/b/**)",
        "Bash",
        "WebFetch",
    ]


def test_explicit_allowed_tools_are_used_as_they_are() -> None:
    arguments = claude_arguments(
        ClaudeCodeConfig(tools=("Read", "Bash"), allowed_tools=("Read", "Bash(ls:*)"))
    )

    assert _value(arguments, "--allowedTools") == "Read,Bash(ls:*)"


def test_no_tools_is_an_explicit_empty_argument() -> None:
    arguments = claude_arguments(ClaudeCodeConfig(tools=()))

    assert _value(arguments, "--tools") == ""
    assert "--allowedTools" not in arguments


def test_configured_options_become_arguments() -> None:
    arguments = claude_arguments(
        ClaudeCodeConfig(
            model="opus",
            effort="high",
            permission_mode="acceptEdits",
            tools=("Read", "Edit", "Bash"),
            allowed_tools=("Read", "Bash(git status:*)"),
            disallowed_tools=("Bash(rm:*)",),
            max_turns=7,
            max_budget_usd=1.5,
            stream_deltas=True,
            mcp_servers={"docs": {"command": "docs-server", "args": ["--fast"]}},
            subagents={"explorer": {"description": "explores", "prompt": "explore"}},
            additional_directories=("/data/a", "/data/b"),
            plugin_directories=("/plugins/p",),
        ),
        instructions="be careful",
    )

    assert _value(arguments, "--model") == "opus"
    assert _value(arguments, "--effort") == "high"
    assert _value(arguments, "--permission-mode") == "acceptEdits"
    assert _value(arguments, "--tools") == "Read,Edit,Bash"
    assert _value(arguments, "--allowedTools") == "Read,Bash(git status:*)"
    assert _value(arguments, "--disallowedTools") == "Bash(rm:*)"
    assert _value(arguments, "--max-turns") == "7"
    assert _value(arguments, "--max-budget-usd") == "1.5"
    assert "--include-partial-messages" in arguments
    assert json.loads(_value(arguments, "--mcp-config")) == {
        "mcpServers": {"docs": {"command": "docs-server", "args": ["--fast"]}}
    }
    assert json.loads(_value(arguments, "--agents")) == {
        "explorer": {"description": "explores", "prompt": "explore"}
    }
    assert [arguments[i + 1] for i, a in enumerate(arguments) if a == "--add-dir"] == [
        "/data/a",
        "/data/b",
    ]
    assert _value(arguments, "--plugin-dir") == "/plugins/p"
    assert "--disable-slash-commands" not in arguments  # plugins bring their own
    assert _value(arguments, "--append-system-prompt") == "be careful"


def test_instructions_replace_the_system_prompt_on_request() -> None:
    arguments = claude_arguments(ClaudeCodeConfig(system_prompt="replace"), instructions="only me")

    assert _value(arguments, "--system-prompt") == "only me"
    assert "--append-system-prompt" not in arguments


@pytest.mark.parametrize(
    ("session", "flag", "value"),
    [
        (ClaudeCodeSession(mode="new", identifier="abc"), "--session-id", "abc"),
        (ClaudeCodeSession(mode="resume", identifier="abc"), "--resume", "abc"),
    ],
)
def test_sessions_are_new_or_resumed(session: ClaudeCodeSession, flag: str, value: str) -> None:
    arguments = claude_arguments(ClaudeCodeConfig(), session=session)

    assert _value(arguments, flag) == value
    assert "--no-session-persistence" not in arguments


def test_session_modes_require_an_identifier() -> None:
    with pytest.raises(ValueError, match="identifier"):
        claude_arguments(ClaudeCodeConfig(), session=ClaudeCodeSession(mode="resume"))


def test_settings_disable_builtin_plugins_and_memory_and_merge_configured_ones() -> None:
    settings = json.loads(
        claude_settings(
            ClaudeCodeConfig(
                auto_compact=False,
                settings={"autoMemoryEnabled": True, "custom": {"nested": ["a"]}},
            )
        )
    )

    assert settings["enabledPlugins"]["cc-plugin-telemetry@builtin"] is False
    assert all(enabled is False for enabled in settings["enabledPlugins"].values())
    assert settings["autoCompactEnabled"] is False
    assert settings["autoMemoryEnabled"] is True  # the configured one wins
    assert settings["custom"] == {"nested": ["a"]}
    assert "autoCompactEnabled" not in json.loads(claude_settings(ClaudeCodeConfig()))


def _environment(config: ClaudeCodeConfig | None = None, **arguments: Any) -> Any:
    return claude_environment(
        config or ClaudeCodeConfig(),
        home="/private/home",
        config_directory="/private/claude",
        temporary_directory="/private/tmp",
        parent={
            "PATH": "/usr/bin",
            "LANG": "en_US.UTF-8",
            "HTTPS_PROXY": "http://proxy",
            "CLAUDECODE": "1",
            "CLAUDE_CODE_ENTRYPOINT": "cli",
            "ANTHROPIC_API_KEY": "parent-secret",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://parent-collector",
            "AWS_SECRET_ACCESS_KEY": "other-secret",
            "CUSTOM": "custom",
        },
        **arguments,
    )


def test_environment_is_built_from_scratch() -> None:
    environment = _environment()

    assert environment["PATH"] == "/usr/bin"
    assert environment["LANG"] == "en_US.UTF-8"
    assert environment["HTTPS_PROXY"] == "http://proxy"
    assert environment["HOME"] == "/private/home"
    assert environment["TMPDIR"] == "/private/tmp"
    assert environment["CLAUDE_CONFIG_DIR"] == "/private/claude"
    for isolating in (
        "DISABLE_AUTOUPDATER",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
        "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS",
        "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS",
    ):
        assert environment[isolating] == "1"

    for leaked in (
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "ANTHROPIC_API_KEY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "AWS_SECRET_ACCESS_KEY",
        "CUSTOM",
    ):
        assert leaked not in environment


def test_environment_inherits_only_requested_names_and_adds_configured_ones() -> None:
    environment = _environment(ClaudeCodeConfig(inherit_env=("CUSTOM",), env={"EXTRA": "1"}))

    assert environment["CUSTOM"] == "custom"
    assert environment["EXTRA"] == "1"


def test_environment_carries_endpoint_and_compaction_tuning() -> None:
    environment = _environment(
        ClaudeCodeConfig(auto_compact_percent=60, auto_compact_window=200_000),
        base_url="https://gateway.example",
    )

    assert environment["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert environment["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "60"
    assert environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "200000"
    assert "ANTHROPIC_BASE_URL" not in _environment()


def test_no_telemetry_unless_configured() -> None:
    environment = _environment(trace_context={"traceparent": "00-aa-bb-01"})

    assert not [name for name in environment if name.startswith("OTEL_")]
    assert "CLAUDE_CODE_ENABLE_TELEMETRY" not in environment
    assert "TRACEPARENT" not in environment


def test_telemetry_is_configured_through_the_environment() -> None:
    environment = _environment(
        ClaudeCodeConfig(
            telemetry=ClaudeCodeTelemetry(
                endpoint="http://collector:4317",
                headers="authorization=Bearer abc",
                traces=True,
                log_prompts=True,
                resource_attributes={"team": "a b"},
            )
        ),
        trace_context={"traceparent": "00-aa-bb-01", "tracestate": "vendor=1"},
        resource_attributes={"draive.agent": "coder"},
    )

    assert environment["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert environment["OTEL_METRICS_EXPORTER"] == "otlp"
    assert environment["OTEL_LOGS_EXPORTER"] == "otlp"
    assert environment["OTEL_TRACES_EXPORTER"] == "otlp"
    assert environment["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == "1"
    assert environment["OTEL_EXPORTER_OTLP_PROTOCOL"] == "grpc"
    assert environment["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"
    assert environment["OTEL_EXPORTER_OTLP_HEADERS"] == "authorization=Bearer abc"
    assert environment["OTEL_METRIC_EXPORT_INTERVAL"] == "5000"
    assert environment["OTEL_LOG_USER_PROMPTS"] == "1"
    assert "OTEL_LOG_TOOL_DETAILS" not in environment
    assert environment["TRACEPARENT"] == "00-aa-bb-01"
    assert environment["TRACESTATE"] == "vendor=1"
    assert environment["OTEL_RESOURCE_ATTRIBUTES"] == (
        "service.name=claude-code,team=a%20b,draive.agent=coder"
    )


def test_telemetry_signals_can_be_disabled_and_the_trace_not_propagated() -> None:
    environment = _environment(
        ClaudeCodeConfig(
            telemetry=ClaudeCodeTelemetry(metrics=False, logs=False, propagate_trace=False)
        ),
        trace_context={"traceparent": "00-aa-bb-01"},
    )

    assert environment["OTEL_METRICS_EXPORTER"] == "none"
    assert environment["OTEL_LOGS_EXPORTER"] == "none"
    assert "OTEL_TRACES_EXPORTER" not in environment
    assert "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA" not in environment
    assert "TRACEPARENT" not in environment
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in environment


def test_telemetry_from_env_reads_the_standard_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "k=v")

    telemetry = ClaudeCodeTelemetry.from_env(traces=True)

    assert telemetry.endpoint == "http://collector:4318"
    assert telemetry.protocol == "http/protobuf"
    assert telemetry.headers == "k=v"
    assert telemetry.traces is True
    assert "k=v" not in repr(telemetry)  # headers carry credentials


def test_telemetry_from_env_without_variables_leaves_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_HEADERS",
    ):
        monkeypatch.delenv(name, raising=False)

    telemetry = ClaudeCodeTelemetry.from_env()

    assert telemetry.endpoint is Missing
    assert telemetry.protocol == "grpc"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (ClaudeCodeConfig(persistence="resumable"), "state_directory"),
        (ClaudeCodeConfig(timeout=0), "timeout"),
        (ClaudeCodeConfig(stop_grace=-1), "stop_grace"),
        (ClaudeCodeConfig(max_turns=0), "max_turns"),
        (ClaudeCodeConfig(max_budget_usd=0), "max_budget_usd"),
        (ClaudeCodeConfig(auto_compact_percent=0), "auto_compact_percent"),
        (ClaudeCodeConfig(auto_compact_percent=101), "auto_compact_percent"),
        (ClaudeCodeConfig(auto_compact_window=1000), "auto_compact_window"),
        (ClaudeCodeConfig(additional_directories=("relative",)), "absolute"),
        (ClaudeCodeConfig(plugin_directories=("relative",)), "absolute"),
        *[
            (ClaudeCodeConfig(env={name: "x"}), "reserved")
            for name in (
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDECODE",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "DISABLE_TELEMETRY",
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "HOME",
                "TMPDIR",
                "TRACEPARENT",
                "DRAIVE_CLAUDE_RUN",
            )
        ],
        (ClaudeCodeConfig(inherit_env=("ANTHROPIC_API_KEY",)), "reserved"),
    ],
)
def test_invalid_configuration_is_refused(config: ClaudeCodeConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_valid_configuration_passes() -> None:
    validate_config(
        ClaudeCodeConfig(
            persistence="resumable",
            state_directory="/state",
            max_turns=1,
            max_budget_usd=0.1,
            auto_compact_percent=50,
            auto_compact_window=100_000,
            env={"FAKE": "1"},
        )
    )


def test_launch_keeps_credentials_out_of_arguments_and_environment(tmp_path: Path) -> None:
    launch = build_launch(
        ClaudeCodeConfig(
            telemetry=ClaudeCodeTelemetry(headers="authorization=Bearer telemetry-secret")
        ),
        binary="/bin/claude",
        prompt=({"type": "text", "text": "hello"},),
        directory=str(tmp_path),
        home=str(tmp_path),
        config_directory=str(tmp_path),
        temporary_directory=str(tmp_path),
        instructions="be brief",
        credential="sk-ant-api03-secret",
    )

    assert "sk-ant-api03-secret" not in " ".join(launch.arguments)
    assert "sk-ant-api03-secret" not in " ".join(launch.environment.values())
    assert launch.credential == "sk-ant-api03-secret"
    assert launch.credential_variable == "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"
    assert launch.secrets == ("sk-ant-api03-secret", "authorization=Bearer telemetry-secret")
    assert "sk-ant-api03-secret" not in repr(launch)
    assert "telemetry-secret" not in repr(launch)
    assert json.loads(launch.prompt) == {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    }
    assert "\n" not in launch.prompt


def test_oauth_credentials_use_their_own_descriptor_variable(tmp_path: Path) -> None:
    launch = build_launch(
        ClaudeCodeConfig(),
        binary="/bin/claude",
        prompt=({"type": "text", "text": "hello"},),
        directory=str(tmp_path),
        home=str(tmp_path),
        config_directory=str(tmp_path),
        temporary_directory=str(tmp_path),
        credential="sk-ant-oat-token",
        credential_kind="oauth_token",
    )

    assert launch.credential_variable == "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"


def test_launch_without_credentials_has_none(tmp_path: Path) -> None:
    launch = build_launch(
        ClaudeCodeConfig(),
        binary="/bin/claude",
        prompt=({"type": "text", "text": "hello"},),
        directory=str(tmp_path),
        home=str(tmp_path),
        config_directory=str(tmp_path),
        temporary_directory=str(tmp_path),
    )

    assert launch.credential is Missing
    assert launch.secrets == ()
