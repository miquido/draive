import json
from pathlib import Path
from typing import Any

import pytest
from haiway import Missing, ValidationError

from draive.claude import (
    ClaudeCodeApiKey,
    ClaudeCodeAutoCompaction,
    ClaudeCodeConfig,
    ClaudeCodeNewSession,
    ClaudeCodeNoCompaction,
    ClaudeCodeNoSandbox,
    ClaudeCodeNoSession,
    ClaudeCodeOAuthToken,
    ClaudeCodeResumedSession,
    ClaudeCodeSandbox,
    ClaudeCodeTelemetry,
)
from draive.claude.launch import ClaudeCodeLaunch
from draive.claude.workspace import ClaudeCodeWorkspace

SESSION = "0a3b6f2e-1c9d-5e4a-8b7f-2d6c4e8a9b01"
WORKSPACE = ClaudeCodeWorkspace(root="/private", directory="/private/work")


def _value(arguments: Any, flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


def test_default_arguments_isolate_and_deny_by_default() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(ClaudeCodeConfig())

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
    assert _value(arguments, "--tools") == ""  # no built-in tools unless configured
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
        "--allowedTools",
    ):
        assert absent not in arguments


def test_file_tools_are_allowed_within_the_directories_of_the_run_only() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(ClaudeCodeConfig(tools=("Read", "Grep", "Glob")))

    assert _value(arguments, "--tools") == "Read,Grep,Glob"
    assert _value(arguments, "--allowedTools") == "Read(./**),Grep(./**),Glob(./**)"


def test_setting_sources_are_empty_unless_configured() -> None:
    assert _value(ClaudeCodeLaunch.prepare_arguments(ClaudeCodeConfig()), "--setting-sources") == ""
    assert (
        _value(
            ClaudeCodeLaunch.prepare_arguments(
                ClaudeCodeConfig(setting_sources=("project", "local"))
            ),
            "--setting-sources",
        )
        == "project,local"
    )


def test_file_tools_are_scoped_to_the_directories_of_the_run_by_default() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
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


def test_tools_changing_files_are_allowed_by_edit_rules() -> None:
    # `Write(path)` rules never match, permissions of every tool changing files are `Edit` ones
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(tools=("Write", "Edit", "NotebookEdit", "Grep", "Glob"))
    )

    assert _value(arguments, "--allowedTools").split(",") == [
        "Edit(./**)",
        "Grep(./**)",
        "Glob(./**)",
    ]


def test_additional_directories_are_escaped_within_rules() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(
            tools=("Read",),
            additional_directories=("/data/project[12]", "/data/a*b?c", "/data/back\\slash"),
        )
    )

    assert _value(arguments, "--allowedTools").split(",") == [
        "Read(./**)",
        "Read(//data/project\\[12\\]/**)",
        "Read(//data/a\\*b\\?c/**)",
        "Read(//data/back\\\\slash/**)",
    ]


def test_resumed_sessions_do_not_keep_the_instructions_of_their_first_request() -> None:
    resumed = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(), session=ClaudeCodeResumedSession(identifier=SESSION)
    )

    assert _value(resumed, "--system-prompt-snapshot") == "off"
    for session in (ClaudeCodeNoSession(), ClaudeCodeNewSession(identifier=SESSION)):
        arguments = ClaudeCodeLaunch.prepare_arguments(ClaudeCodeConfig(), session=session)
        assert "--system-prompt-snapshot" not in arguments


def test_explicit_allowed_tools_are_used_as_they_are() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(tools=("Read", "Bash"), allowed_tools=("Read", "Bash(ls:*)"))
    )

    assert _value(arguments, "--allowedTools") == "Read,Bash(ls:*)"


def test_no_tools_is_an_explicit_empty_argument() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(ClaudeCodeConfig(tools=()))

    assert _value(arguments, "--tools") == ""
    assert "--allowedTools" not in arguments


def test_configured_options_become_arguments() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
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
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(system_prompt="replace"), instructions="only me"
    )

    assert _value(arguments, "--system-prompt") == "only me"
    assert "--append-system-prompt" not in arguments


def test_new_sessions_are_started_by_their_identifier() -> None:
    session = ClaudeCodeNewSession(identifier=SESSION)
    arguments = ClaudeCodeLaunch.prepare_arguments(ClaudeCodeConfig(), session=session)

    assert _value(arguments, "--session-id") == SESSION
    assert "--no-session-persistence" not in arguments
    assert "--resume" not in arguments


def test_resumed_sessions_are_continued_by_their_identifier() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(), session=ClaudeCodeResumedSession(identifier=SESSION)
    )

    assert _value(arguments, "--resume") == SESSION
    assert "--no-session-persistence" not in arguments
    assert "--session-id" not in arguments


def test_runs_without_a_session_leave_none() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(), session=ClaudeCodeNoSession()
    )

    assert "--no-session-persistence" in arguments
    assert "--session-id" not in arguments
    assert "--resume" not in arguments


@pytest.mark.parametrize("identifier", ["", "abc", "not-a-uuid", " "])
@pytest.mark.parametrize("session", [ClaudeCodeNewSession, ClaudeCodeResumedSession])
def test_sessions_without_a_valid_identifier_can_not_exist(session: Any, identifier: str) -> None:
    with pytest.raises(ValidationError):
        session(identifier=identifier)


def test_settings_disable_builtin_plugins_and_memory_and_merge_configured_ones() -> None:
    settings = json.loads(
        ClaudeCodeLaunch.prepare_settings(
            ClaudeCodeConfig(
                compaction=ClaudeCodeNoCompaction(),
                settings={"autoMemoryEnabled": True, "custom": {"nested": ["a"]}},
            )
        )
    )

    assert settings["enabledPlugins"]["cc-plugin-telemetry@builtin"] is False
    assert all(enabled is False for enabled in settings["enabledPlugins"].values())
    assert settings["autoCompactEnabled"] is False
    assert settings["autoMemoryEnabled"] is True  # the configured one wins
    assert settings["custom"] == {"nested": ["a"]}
    assert "autoCompactEnabled" not in json.loads(
        ClaudeCodeLaunch.prepare_settings(ClaudeCodeConfig())
    )


def _user_memory(home: Path) -> list[str]:
    resolved = home.resolve()
    return [
        f"{resolved}/CLAUDE.md",
        f"{resolved}/CLAUDE.local.md",
        f"{resolved}/.claude/CLAUDE.md",
        f"{resolved}/.claude/**",
    ]


def test_settings_exclude_the_claude_md_of_the_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # written as they are, an escaped character would keep the exclusion from matching
    home = tmp_path / "us[e]r(1)"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    settings = json.loads(
        ClaudeCodeLaunch.prepare_settings(ClaudeCodeConfig(setting_sources=("project",)))
    )

    assert settings["claudeMdExcludes"] == _user_memory(home)


def test_configured_claude_md_excludes_extend_the_isolating_ones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    settings = json.loads(
        ClaudeCodeLaunch.prepare_settings(
            ClaudeCodeConfig(settings={"claudeMdExcludes": ["/repo/vendor/**"]})
        )
    )

    assert settings["claudeMdExcludes"] == [*_user_memory(tmp_path), "/repo/vendor/**"]


def test_builtin_plugins_unknown_to_the_settings_are_leaks() -> None:
    plugins = ("cc-plugin-tips", "user-plugin", "cc-plugin-new")
    config = ClaudeCodeConfig()

    assert ClaudeCodeLaunch.leaked_plugins(config, plugins) == ("cc-plugin-new",)
    # disabled and loaded anyway, enforced by the organization of the account
    assert ClaudeCodeLaunch.enforced_plugins(config, plugins) == ("cc-plugin-tips",)


def test_plugins_enabled_by_the_settings_are_neither_leaked_nor_enforced() -> None:
    plugins = ("cc-plugin-diff", "cc-plugin-tips", "cc-plugin-new")
    config = ClaudeCodeConfig(settings={"enabledPlugins": {"cc-plugin-diff@builtin": True}})

    assert ClaudeCodeLaunch.leaked_plugins(config, plugins) == ("cc-plugin-new",)
    assert ClaudeCodeLaunch.enforced_plugins(config, plugins) == ("cc-plugin-tips",)


def test_plugins_known_to_the_settings_are_not_leaked() -> None:
    config = ClaudeCodeConfig(settings={"enabledPlugins": {"cc-plugin-new@builtin": False}})

    assert ClaudeCodeLaunch.leaked_plugins(config, ("cc-plugin-new",)) == ()
    assert ClaudeCodeLaunch.enforced_plugins(config, ("cc-plugin-new",)) == ("cc-plugin-new",)


@pytest.mark.parametrize(
    "settings",
    (
        {"enabledPlugins": None},
        {"enabledPlugins": ["cc-plugin-diff@builtin"]},
        {"enabledPlugins": {"mine@local": "yes"}},
        {"claudeMdExcludes": None},
        {"claudeMdExcludes": "/repo/**"},
        {"claudeMdExcludes": ["/repo/**", 1]},
        {"sandbox": ["enabled"]},
        {"sandbox": True},
    ),
)
def test_wrong_typed_isolating_settings_are_refused(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ClaudeCodeConfig(settings=settings)


def test_configured_plugins_extend_the_disabled_ones() -> None:
    settings = json.loads(
        ClaudeCodeLaunch.prepare_settings(
            ClaudeCodeConfig(
                settings={"enabledPlugins": {"mine@local": True, "cc-plugin-diff@builtin": True}}
            )
        )
    )

    assert settings["enabledPlugins"]["mine@local"] is True
    assert settings["enabledPlugins"]["cc-plugin-diff@builtin"] is True  # enabled explicitly
    assert settings["enabledPlugins"]["cc-plugin-tips@builtin"] is False


SANDBOX = {
    "enabled": True,
    "allowUnsandboxedCommands": False,
    "failIfUnavailable": True,
}


def test_bash_is_sandboxed_strictly_by_default() -> None:
    settings = json.loads(ClaudeCodeLaunch.prepare_settings(ClaudeCodeConfig(tools=("Bash",))))

    assert settings["sandbox"] == SANDBOX


@pytest.mark.parametrize(
    "config",
    (
        ClaudeCodeConfig(),
        ClaudeCodeConfig(tools=("Read", "Edit")),
        ClaudeCodeConfig(tools=("Bash",), sandbox=ClaudeCodeNoSandbox()),
    ),
)
def test_no_sandbox_is_set_without_bash_or_when_opted_out(config: ClaudeCodeConfig) -> None:
    assert "sandbox" not in json.loads(ClaudeCodeLaunch.prepare_settings(config))


def _sandbox(sandbox: ClaudeCodeSandbox) -> Any:
    return json.loads(
        ClaudeCodeLaunch.prepare_settings(ClaudeCodeConfig(tools=("Bash",), sandbox=sandbox))
    )["sandbox"]


def test_default_sandbox_is_minimal() -> None:
    sandbox = _sandbox(ClaudeCodeSandbox())

    assert "autoAllowBashIfSandboxed" not in sandbox
    assert not {"filesystem", "excludedCommands", "network"} & sandbox.keys()


@pytest.mark.parametrize(
    ("fields", "expected"),
    (
        ({"allowed_writes": ("/data",)}, {"filesystem": {"allowWrite": ["/data"]}}),
        ({"denied_writes": ("/data",)}, {"filesystem": {"denyWrite": ["/data"]}}),
        ({"denied_reads": ("/secret",)}, {"filesystem": {"denyRead": ["/secret"]}}),
        ({"allowed_reads": ("/secret/ok",)}, {"filesystem": {"allowRead": ["/secret/ok"]}}),
        ({"excluded_commands": ("docker:*",)}, {"excludedCommands": ["docker:*"]}),
        ({"allowed_domains": ("*.x.com",)}, {"network": {"allowedDomains": ["*.x.com"]}}),
        ({"denied_domains": ("evil.com",)}, {"network": {"deniedDomains": ["evil.com"]}}),
    ),
)
def test_sandbox_field_maps_to_its_key(fields: dict[str, Any], expected: dict[str, Any]) -> None:
    assert _sandbox(ClaudeCodeSandbox(**fields)) == {**SANDBOX, **expected}


@pytest.mark.parametrize(
    "field",
    ("allowed_writes", "denied_writes", "denied_reads", "allowed_reads"),
)
def test_relative_sandbox_paths_are_refused(field: str) -> None:
    with pytest.raises(ValidationError):
        ClaudeCodeSandbox(**{field: ("relative/path",)})


def test_configured_sandbox_lists_are_added_to_the_derived_ones() -> None:
    settings = json.loads(
        ClaudeCodeLaunch.prepare_settings(
            ClaudeCodeConfig(
                tools=("Bash",),
                sandbox=ClaudeCodeSandbox(
                    allowed_writes=("/data",),
                    denied_reads=("/secret",),
                    allowed_domains=("example.com",),
                ),
                settings={
                    "sandbox": {
                        "filesystem": {"allowWrite": ["/data", "/other"]},
                        "network": {"allowedDomains": ["other.com"], "allowLocalBinding": True},
                    }
                },
            )
        )
    )

    assert settings["sandbox"]["filesystem"] == {
        "allowWrite": ["/data", "/other"],
        "denyRead": ["/secret"],
    }
    assert settings["sandbox"]["network"] == {
        "allowedDomains": ["example.com", "other.com"],
        "allowLocalBinding": True,
    }


def test_configured_sandbox_sections_are_passed_when_nothing_is_derived() -> None:
    settings = json.loads(
        ClaudeCodeLaunch.prepare_settings(
            ClaudeCodeConfig(
                tools=("Bash",),
                settings={"sandbox": {"filesystem": {"allowWrite": ["/data"]}}},
            )
        )
    )

    assert settings["sandbox"]["filesystem"] == {"allowWrite": ["/data"]}


def test_configured_sandbox_keys_are_merged_over_the_derived_ones() -> None:
    settings = json.loads(
        ClaudeCodeLaunch.prepare_settings(
            ClaudeCodeConfig(
                tools=("Bash",),
                settings={"sandbox": {"network": {"allowedDomains": ["example.com"]}}},
            )
        )
    )

    assert settings["sandbox"] == {**SANDBOX, "network": {"allowedDomains": ["example.com"]}}


def test_configured_sandbox_keys_win_over_the_derived_ones() -> None:
    settings = json.loads(
        ClaudeCodeLaunch.prepare_settings(
            ClaudeCodeConfig(tools=("Bash",), settings={"sandbox": {"enabled": False}})
        )
    )

    assert settings["sandbox"] == {**SANDBOX, "enabled": False}


def test_configured_sandbox_is_passed_without_bash() -> None:
    settings = json.loads(
        ClaudeCodeLaunch.prepare_settings(ClaudeCodeConfig(settings={"sandbox": {"enabled": True}}))
    )

    assert settings["sandbox"] == {"enabled": True}


def _workspace(directory: Path) -> ClaudeCodeWorkspace:
    return ClaudeCodeWorkspace(root=str(directory.parent), directory=str(directory))


@pytest.mark.parametrize(
    ("layout", "reaches"),
    [
        pytest.param("plain", True, id="within_home"),
        pytest.param("repository", False, id="within_repository"),
        pytest.param("home_repository", True, id="home_is_repository"),
        pytest.param("no_agents", False, id="user_without_agents"),
        pytest.param("outside", False, id="outside_home"),
    ],
)
@pytest.mark.asyncio
async def test_workspace_knows_when_project_settings_reach_user_agents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    reaches: bool,
) -> None:
    home = tmp_path / "home"
    project = home / "projects" / "app"
    project.mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setenv("HOME", str(home))
    if layout != "no_agents":
        (home / ".claude" / "agents").mkdir(parents=True)

    if layout == "repository":
        (home / "projects" / ".git").mkdir()

    if layout == "home_repository":
        (home / ".git").mkdir()

    directory = outside if layout == "outside" else project
    assert await _workspace(directory).reaches_user_agents() is reaches


def _environment(config: ClaudeCodeConfig | None = None, **arguments: Any) -> Any:
    return ClaudeCodeLaunch.prepare_environment(
        config or ClaudeCodeConfig(),
        workspace=WORKSPACE,
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
        ClaudeCodeConfig(compaction=ClaudeCodeAutoCompaction(percent=60, window=200_000)),
        base_url="https://gateway.example",
    )

    assert environment["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert environment["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "60"
    assert environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "200000"
    assert "ANTHROPIC_BASE_URL" not in _environment()


def test_compaction_is_left_to_claude_code_unless_tuned_or_disabled() -> None:
    untouched = _environment()
    disabled = _environment(ClaudeCodeConfig(compaction=ClaudeCodeNoCompaction()))

    for environment in (untouched, disabled):
        assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in environment
        assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in environment


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
                signals={"metrics", "logs", "traces"},
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
    for interval in (
        "OTEL_METRIC_EXPORT_INTERVAL",
        "OTEL_LOGS_EXPORT_INTERVAL",
        "OTEL_TRACES_EXPORT_INTERVAL",
    ):
        assert environment[interval] == "2000"
    assert environment["OTEL_LOG_USER_PROMPTS"] == "1"
    assert "OTEL_LOG_TOOL_DETAILS" not in environment
    assert environment["TRACEPARENT"] == "00-aa-bb-01"
    assert environment["TRACESTATE"] == "vendor=1"
    assert environment["OTEL_RESOURCE_ATTRIBUTES"] == (
        "service.name=claude-code,team=a%20b,draive.agent=coder"
    )


def test_telemetry_exports_metrics_and_events_unless_told_otherwise() -> None:
    environment = _environment(ClaudeCodeConfig(telemetry=ClaudeCodeTelemetry()))

    assert environment["OTEL_METRICS_EXPORTER"] == "otlp"
    assert environment["OTEL_LOGS_EXPORTER"] == "otlp"
    assert "OTEL_TRACES_EXPORTER" not in environment  # a beta of Claude Code
    assert "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA" not in environment
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in environment


def test_telemetry_exports_only_the_signals_it_is_given() -> None:
    environment = _environment(
        ClaudeCodeConfig(
            telemetry=ClaudeCodeTelemetry(signals={"traces"}, export_interval_ms=500),
        ),
    )

    assert environment["OTEL_METRICS_EXPORTER"] == "none"
    assert environment["OTEL_LOGS_EXPORTER"] == "none"
    assert environment["OTEL_TRACES_EXPORTER"] == "otlp"
    assert environment["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == "1"
    assert environment["OTEL_TRACES_EXPORT_INTERVAL"] == "500"


def test_telemetry_continues_the_trace_unless_told_otherwise() -> None:
    context = {"traceparent": "00-aa-bb-01"}

    continued = _environment(
        ClaudeCodeConfig(telemetry=ClaudeCodeTelemetry()),
        trace_context=context,
    )
    alone = _environment(
        ClaudeCodeConfig(telemetry=ClaudeCodeTelemetry(propagate_trace=False)),
        trace_context=context,
    )

    assert continued["TRACEPARENT"] == "00-aa-bb-01"
    assert "TRACEPARENT" not in alone


def test_the_name_of_the_service_can_be_replaced_by_its_attributes() -> None:
    default = _environment(ClaudeCodeConfig(telemetry=ClaudeCodeTelemetry()))
    renamed = _environment(
        ClaudeCodeConfig(
            telemetry=ClaudeCodeTelemetry(resource_attributes={"service.name": "my-agents"})
        )
    )

    assert default["OTEL_RESOURCE_ATTRIBUTES"] == "service.name=claude-code"
    assert renamed["OTEL_RESOURCE_ATTRIBUTES"] == "service.name=my-agents"


@pytest.mark.parametrize("signals", [frozenset(), set()])
def test_telemetry_without_a_signal_can_not_exist(signals: Any) -> None:
    with pytest.raises(ValidationError):
        ClaudeCodeTelemetry(signals=signals)


def test_telemetry_from_env_reads_the_standard_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "k=v")

    telemetry = ClaudeCodeTelemetry.from_env(signals={"traces"})

    assert telemetry.endpoint == "http://collector:4318"
    assert telemetry.protocol == "http/protobuf"
    assert telemetry.headers == "k=v"
    assert telemetry.signals == {"traces"}
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
    "make",
    [
        lambda: ClaudeCodeConfig(timeout=0),
        lambda: ClaudeCodeConfig(stop_grace=-1),
        lambda: ClaudeCodeConfig(max_turns=0),
        lambda: ClaudeCodeConfig(max_budget_usd=0),
        lambda: ClaudeCodeAutoCompaction(percent=0),
        lambda: ClaudeCodeAutoCompaction(percent=101),
        lambda: ClaudeCodeAutoCompaction(window=1000),
        lambda: ClaudeCodeConfig(additional_directories=("relative",)),
        lambda: ClaudeCodeConfig(plugin_directories=("relative",)),
        lambda: ClaudeCodeConfig(directory="relative"),
        lambda: ClaudeCodeTelemetry(export_interval_ms=0),
        lambda: ClaudeCodeConfig(inherit_env=("ANTHROPIC_API_KEY",)),
        lambda: ClaudeCodeConfig(inherit_env=("MCP_CONNECTION_NONBLOCKING",)),
        *[
            (lambda name=name: ClaudeCodeConfig(env={name: "x"}))
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
                "MCP_CONNECTION_NONBLOCKING",
                "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD",
            )
        ],
    ],
)
def test_invalid_configuration_can_not_be_created(make: Any) -> None:
    with pytest.raises(ValidationError):
        make()


def test_valid_configuration_is_created() -> None:
    ClaudeCodeConfig(
        max_turns=1,
        max_budget_usd=0.1,
        compaction=ClaudeCodeAutoCompaction(percent=50, window=100_000),
        env={"FAKE": "1"},
        directory="/project",
        additional_directories=("/data",),
    )


@pytest.mark.parametrize(
    "config",
    [
        ClaudeCodeConfig(),
        ClaudeCodeConfig(compaction=ClaudeCodeNoCompaction()),
        ClaudeCodeConfig(compaction=ClaudeCodeAutoCompaction(percent=40)),
        ClaudeCodeConfig(telemetry=ClaudeCodeTelemetry(signals={"traces", "logs"})),
    ],
)
def test_configuration_survives_serialization(config: ClaudeCodeConfig) -> None:
    # variants of a sum type are told apart by their tag
    assert ClaudeCodeConfig.from_mapping(config.to_mapping()) == config


def test_variants_of_compaction_are_distinct_after_serialization() -> None:
    disabled = ClaudeCodeConfig(compaction=ClaudeCodeNoCompaction())

    restored = ClaudeCodeConfig.from_mapping(disabled.to_mapping())

    assert isinstance(restored.compaction, ClaudeCodeNoCompaction)


def _launch(**arguments: Any) -> Any:
    return ClaudeCodeLaunch.of(
        arguments.pop("config", ClaudeCodeConfig()),
        binary="/bin/claude",
        prompt=({"type": "text", "text": "hello"},),
        workspace=WORKSPACE,
        **arguments,
    )


def test_launch_keeps_credentials_out_of_arguments_and_environment() -> None:
    launch = _launch(
        config=ClaudeCodeConfig(
            telemetry=ClaudeCodeTelemetry(headers="authorization=Bearer telemetry-secret")
        ),
        instructions="be brief",
        credential=ClaudeCodeApiKey(value="sk-ant-api03-secret"),
    )

    assert "sk-ant-api03-secret" not in " ".join(launch.arguments)
    assert "sk-ant-api03-secret" not in " ".join(launch.environment.values())
    assert launch.credential == ClaudeCodeApiKey(value="sk-ant-api03-secret")
    assert "sk-ant-api03-secret" not in repr(launch)
    assert "telemetry-secret" not in repr(launch)
    assert json.loads(launch.prompt) == {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    }
    assert "\n" not in launch.prompt


def test_what_is_redacted_is_derived_from_the_credential_and_the_environment() -> None:
    launch = _launch(
        config=ClaudeCodeConfig(
            telemetry=ClaudeCodeTelemetry(headers="authorization=Bearer telemetry-secret")
        ),
        credential=ClaudeCodeApiKey(value="sk-ant-api03-secret"),
    )

    assert launch.redactions == ("sk-ant-api03-secret", "authorization=Bearer telemetry-secret")
    assert _launch().redactions == ()


def test_launch_carries_neither_the_marker_nor_the_descriptor_in_its_environment() -> None:
    # added by the run from the launch, which is their only source
    launch = _launch(credential=ClaudeCodeOAuthToken(value="sk-ant-oat-token"))

    assert "DRAIVE_CLAUDE_RUN" not in launch.environment
    assert not [name for name in launch.environment if name.endswith("_FILE_DESCRIPTOR")]
    assert launch.marker


def test_every_launch_has_its_own_marker() -> None:
    assert _launch().marker != _launch().marker


def test_launch_without_credentials_has_none() -> None:
    assert _launch().credential is Missing


def test_credentials_describe_how_claude_code_receives_and_reports_them() -> None:
    key = ClaudeCodeApiKey(value="sk-ant-api03-k")
    token = ClaudeCodeOAuthToken(value="sk-ant-oat-t")

    assert key.descriptor == "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"
    assert token.descriptor == "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"
    assert key.reported_source == "ANTHROPIC_API_KEY"
    assert token.reported_source == "none"


@pytest.mark.parametrize("credential", [ClaudeCodeApiKey, ClaudeCodeOAuthToken])
@pytest.mark.parametrize("value", ["", "k" * 4097])
def test_credentials_which_can_not_be_handed_over_can_not_exist(
    credential: Any,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        credential(value=value)


@pytest.mark.parametrize("credential", [ClaudeCodeApiKey, ClaudeCodeOAuthToken])
def test_credentials_never_show_their_value(credential: Any) -> None:
    assert "very-secret" not in repr(credential(value="very-secret"))
    assert "very-secret" not in str(credential(value="very-secret"))


_TOOLS_SERVER = {"command": "/usr/bin/python", "args": ["-I", "/relay.py", "/tmp/mcp.sock"]}


def test_tools_server_is_added_to_the_mcp_config_and_allowed() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(mcp_servers={"docs": {"command": "docs-server"}}),
        tools_server=_TOOLS_SERVER,
    )

    assert json.loads(_value(arguments, "--mcp-config")) == {
        "mcpServers": {"docs": {"command": "docs-server"}, "draive": _TOOLS_SERVER}
    }
    assert _value(arguments, "--allowedTools") == "mcp__draive"


def test_tools_server_is_allowed_next_to_scoped_file_tools() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(tools=("Read", "Grep", "Glob")),
        tools_server=_TOOLS_SERVER,
    )

    assert _value(arguments, "--allowedTools") == "Read(./**),Grep(./**),Glob(./**),mcp__draive"


def test_tools_server_is_allowed_next_to_explicit_allowed_tools() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(
            tools=("Read", "Bash"),
            allowed_tools=("Read", "Bash(ls:*)"),
            disallowed_tools=("Bash(rm:*)",),
        ),
        tools_server=_TOOLS_SERVER,
    )

    assert _value(arguments, "--allowedTools") == "Read,Bash(ls:*),mcp__draive"
    assert _value(arguments, "--disallowedTools") == "Bash(rm:*)"
    assert json.loads(_value(arguments, "--mcp-config")) == {
        "mcpServers": {"draive": _TOOLS_SERVER}
    }


def test_tools_server_is_allowed_without_built_in_tools() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(tools=()),
        tools_server=_TOOLS_SERVER,
    )

    assert _value(arguments, "--allowedTools") == "mcp__draive"


def test_tools_server_name_is_reserved() -> None:
    with pytest.raises(ValueError, match="reserved"):
        ClaudeCodeLaunch.prepare_arguments(
            ClaudeCodeConfig(mcp_servers={"draive": {"command": "other"}}),
            tools_server=_TOOLS_SERVER,
        )


def test_configured_draive_server_is_kept_without_tools_server() -> None:
    arguments = ClaudeCodeLaunch.prepare_arguments(
        ClaudeCodeConfig(mcp_servers={"draive": {"command": "other"}}),
    )

    assert json.loads(_value(arguments, "--mcp-config")) == {
        "mcpServers": {"draive": {"command": "other"}}
    }
    assert "--allowedTools" not in arguments
