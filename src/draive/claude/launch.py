import json
import os
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Final, Literal, cast, final
from urllib.parse import quote
from uuid import uuid4

from haiway import Missing, Sensitive, State

from draive.claude.config import ClaudeCodeConfig, ClaudeCodeTelemetry

__all__ = (
    "RUN_MARKER_VARIABLE",
    "ClaudeCodeLaunch",
    "ClaudeCodeSession",
    "build_launch",
    "claude_arguments",
    "claude_environment",
    "claude_settings",
    "validate_config",
)

RUN_MARKER_VARIABLE: Final[str] = "DRAIVE_CLAUDE_RUN"

# Claude Code registers these plugins on its own - each is disabled, a later version might add
# more which are reported by the started run
_BUILTIN_PLUGINS: Final[Sequence[str]] = (
    "cc-plugin-agents-md@builtin",
    "cc-plugin-claude-platform@builtin",
    "cc-plugin-claude-test@builtin",
    "cc-plugin-diff@builtin",
    "cc-plugin-easel@builtin",
    "cc-plugin-mermaid@builtin",
    "cc-plugin-mods-guide@builtin",
    "cc-plugin-plugin-authoring@builtin",
    "cc-plugin-privacy@builtin",
    "cc-plugin-responsive-mode@builtin",
    "cc-plugin-sec-default@builtin",
    "cc-plugin-telemetry@builtin",
    "cc-plugin-tips@builtin",
    "cc-plugin-you-should-know@builtin",
)

# the process gets nothing of the parent environment but these, anything else of it, i.e. the
# markers of Claude Code running the parent, would change the behavior of the run
_INHERITED_ENV: Final[Sequence[str]] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
)

_RESERVED_ENV_PREFIXES: Final[Sequence[str]] = ("CLAUDE", "ANTHROPIC_", "DISABLE_", "OTEL_")
_RESERVED_ENV_NAMES: Final[Sequence[str]] = (
    RUN_MARKER_VARIABLE,
    "HOME",
    "TMPDIR",
    "TRACEPARENT",
    "TRACESTATE",
    "DO_NOT_TRACK",
    "NODE_OPTIONS",
)


@final
class ClaudeCodeSession(State):
    """Persistence of the run within Claude Code sessions.

    Attributes
    ----------
    mode
        ``ephemeral`` leaves no session, ``new`` starts the session with the given identifier,
        ``resume`` continues the stored one.
    identifier
        Session identifier of the ``new`` and ``resume`` modes.
    """

    mode: Literal["ephemeral", "new", "resume"] = "ephemeral"
    identifier: str | None = None


@final
class ClaudeCodeLaunch(State):
    """Everything needed to run a Claude Code process.

    Attributes
    ----------
    binary
        Absolute path of the executable.
    arguments
        Arguments of the executable, no credentials among them.
    environment
        The whole environment of the process, no credentials among it. Values such as
        telemetry headers can still be sensitive, it is never a part of the representation.
    directory
        Working directory of the process.
    prompt
        The ``user`` message sent to the process within a single ``stream-json`` line.
    credential
        API key or OAuth token, handed over through a file descriptor only.
    credential_variable
        Variable pointing the process to the descriptor of the credential.
    marker
        Value of ``DRAIVE_CLAUDE_RUN`` within the environment of the process and of everything it
        starts - also of what detaches or outlives it, so a run can stop all of it.
    secrets
        Values redacted from anything reported out of the process output.
    timeout
        Limit of the run duration in seconds.
    grace
        Time a stopped process has to exit before it is killed, in seconds.
    """

    binary: str
    arguments: Sequence[str]
    environment: Annotated[Mapping[str, str], Sensitive()]
    directory: str
    prompt: str
    credential: Annotated[str, Sensitive()] | Missing = Missing
    credential_variable: str = "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"
    marker: str = ""
    secrets: Annotated[Sequence[str], Sensitive()] = ()
    timeout: float = 1800.0
    grace: float = 10.0


def validate_config(
    config: ClaudeCodeConfig,
    /,
) -> None:
    """Verify the configuration can be used to run Claude Code.

    Parameters
    ----------
    config
        Configuration to verify.

    Raises
    ------
    ValueError
        When the configuration is invalid, i.e. it uses reserved environment variables.
    """
    if config.persistence == "resumable" and config.state_directory is Missing:
        raise ValueError("Resumable Claude Code runs require state_directory")

    _validate_limits(config)
    for name in (*config.env, *config.inherit_env):
        if _reserved(name):
            raise ValueError(f"Claude Code environment variable '{name}' is reserved")

    for directory in (*config.additional_directories, *config.plugin_directories):
        if not os.path.isabs(directory):
            raise ValueError(f"Claude Code directory '{directory}' has to be absolute")


def _validate_limits(
    config: ClaudeCodeConfig,
    /,
) -> None:
    if config.timeout <= 0 or config.stop_grace < 0:
        raise ValueError("Claude Code timeout has to be positive and stop_grace not negative")

    if config.max_turns is not Missing and config.max_turns < 1:
        raise ValueError("Claude Code max_turns has to be at least 1")

    if config.max_budget_usd is not Missing and config.max_budget_usd <= 0:
        raise ValueError("Claude Code max_budget_usd has to be positive")

    if config.auto_compact_percent is not Missing and not 1 <= config.auto_compact_percent <= 100:  # noqa: PLR2004
        raise ValueError("Claude Code auto_compact_percent has to be between 1 and 100")

    if config.auto_compact_window is not Missing and config.auto_compact_window < 100_000:  # noqa: PLR2004
        raise ValueError("Claude Code auto_compact_window has to be at least 100000 tokens")


def _reserved(
    name: str,
    /,
) -> bool:
    return name in _RESERVED_ENV_NAMES or name.startswith(_RESERVED_ENV_PREFIXES)


def claude_settings(
    config: ClaudeCodeConfig,
    /,
) -> str:
    """Prepare the isolating ``--settings`` JSON, configured settings are merged over it."""
    settings: dict[str, Any] = {
        "enabledPlugins": dict.fromkeys(_BUILTIN_PLUGINS, False),
        "autoMemoryEnabled": False,
    }
    if not config.auto_compact:
        settings["autoCompactEnabled"] = False

    settings.update(config.settings)
    return json.dumps(settings, separators=(",", ":"), default=_json_value)


def _json_value(
    value: object,
    /,
) -> object:
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, Any], value))

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(cast(Sequence[Any], value))

    raise TypeError(f"Unsupported setting value: {type(value).__name__}")


def claude_arguments(  # noqa: C901, PLR0912
    config: ClaudeCodeConfig,
    /,
    *,
    instructions: str = "",
    session: ClaudeCodeSession = ClaudeCodeSession(),  # noqa: B008
) -> Sequence[str]:
    """Prepare the arguments running Claude Code headless within the given configuration.

    Parameters
    ----------
    config
        Configuration of the run.
    instructions
        Instructions of the agent, added to or replacing the system prompt.
    session
        Persistence of the run within Claude Code sessions.

    Returns
    -------
    Sequence[str]
        Arguments of the executable. The prompt is not among them, it is sent to the input.
    """
    arguments: list[str] = [
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--setting-sources",
        ",".join(config.setting_sources),
        "--settings",
        claude_settings(config),
        "--strict-mcp-config",
        "--no-chrome",
        "--permission-prompts",
        "none",
        "--permission-mode",
        config.permission_mode,
        "--model",
        config.model,
        "--tools",
        ",".join(config.tools),
    ]
    if not config.plugin_directories:
        arguments.append("--disable-slash-commands")

    for plugin in config.plugin_directories:
        arguments.extend(("--plugin-dir", plugin))

    allowed: Sequence[str] = (
        _scoped_tools(config) if config.allowed_tools is Missing else config.allowed_tools
    )
    if allowed:
        arguments.extend(("--allowedTools", ",".join(allowed)))

    if config.disallowed_tools:
        arguments.extend(("--disallowedTools", ",".join(config.disallowed_tools)))

    if config.mcp_servers:
        arguments.extend(
            (
                "--mcp-config",
                json.dumps(
                    {"mcpServers": config.mcp_servers},
                    separators=(",", ":"),
                    default=_json_value,
                ),
            )
        )

    if config.subagents:
        arguments.extend(
            (
                "--agents",
                json.dumps(config.subagents, separators=(",", ":"), default=_json_value),
            )
        )

    for directory in config.additional_directories:
        arguments.extend(("--add-dir", directory))

    match session.mode:
        case "ephemeral":
            arguments.append("--no-session-persistence")

        case "new":
            arguments.extend(("--session-id", _session_identifier(session)))

        case "resume":
            arguments.extend(("--resume", _session_identifier(session)))

    if instructions:
        flag: str = (
            "--system-prompt" if config.system_prompt == "replace" else "--append-system-prompt"
        )
        arguments.extend((flag, instructions))

    if config.max_turns is not Missing:
        arguments.extend(("--max-turns", str(config.max_turns)))

    if config.max_budget_usd is not Missing:
        arguments.extend(("--max-budget-usd", str(config.max_budget_usd)))

    if config.effort is not Missing:
        arguments.extend(("--effort", config.effort))

    if config.stream_deltas:
        arguments.append("--include-partial-messages")

    return tuple(arguments)


# tools working with paths, allowed within the directories of the run only - their bare names would
# allow them to work with any path the process can reach, i.e. to read the keys of the user
_PATH_TOOLS: Final[Sequence[str]] = (
    "Read",
    "Grep",
    "Glob",
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
)


def _scoped_tools(
    config: ClaudeCodeConfig,
    /,
) -> Sequence[str]:
    rules: list[str] = []
    for tool in config.tools:
        if tool not in _PATH_TOOLS:
            rules.append(tool)
            continue

        rules.append(f"{tool}(./**)")  # the working directory
        # absolute paths are written with a double slash within the rules
        rules.extend(
            f"{tool}(//{directory.lstrip('/')}/**)" for directory in config.additional_directories
        )

    return tuple(rules)


def _session_identifier(
    session: ClaudeCodeSession,
    /,
) -> str:
    if not session.identifier:
        raise ValueError(f"Claude Code session mode '{session.mode}' requires an identifier")

    return session.identifier


def claude_environment(
    config: ClaudeCodeConfig,
    /,
    *,
    home: str,
    config_directory: str,
    temporary_directory: str,
    base_url: str | None = None,
    trace_context: Mapping[str, str] | None = None,
    resource_attributes: Mapping[str, str] | None = None,
    parent: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Prepare the whole environment of the process, nothing is taken from the parent besides
    a few connectivity variables.

    Parameters
    ----------
    config
        Configuration of the run.
    home
        Private home directory of the process.
    config_directory
        Directory of the Claude Code configuration, it holds sessions of resumable runs.
    temporary_directory
        Directory of temporary files.
    base_url
        Custom API endpoint, i.e. a gateway.
    trace_context
        Trace context of the executing agent, ``traceparent`` and ``tracestate``.
    resource_attributes
        Resource attributes of the exported telemetry in addition to the configured ones.
    parent
        Environment inherited from, the one of the process by default.

    Returns
    -------
    Mapping[str, str]
        Environment of the process.
    """
    source: Mapping[str, str] = os.environ if parent is None else parent
    environment: dict[str, str] = {
        name: source[name] for name in (*_INHERITED_ENV, *config.inherit_env) if name in source
    }
    environment.update(
        {
            "HOME": home,
            "TMPDIR": temporary_directory,
            "CLAUDE_CONFIG_DIR": config_directory,
            "CLAUDE_CODE_TMPDIR": temporary_directory,
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
            "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
        }
    )

    if base_url:
        environment["ANTHROPIC_BASE_URL"] = base_url

    if config.auto_compact_percent is not Missing:
        environment["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(config.auto_compact_percent)

    if config.auto_compact_window is not Missing:
        environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(config.auto_compact_window)

    if config.telemetry is not Missing:
        environment.update(_telemetry_environment(config.telemetry, resource_attributes))
        if config.telemetry.propagate_trace and trace_context:
            if traceparent := trace_context.get("traceparent"):
                environment["TRACEPARENT"] = traceparent

            if tracestate := trace_context.get("tracestate"):
                environment["TRACESTATE"] = tracestate

    environment.update(config.env)
    return environment


def _telemetry_environment(
    telemetry: ClaudeCodeTelemetry,
    resource_attributes: Mapping[str, str] | None,
    /,
) -> Mapping[str, str]:
    environment: dict[str, str] = {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_METRICS_EXPORTER": "otlp" if telemetry.metrics else "none",
        "OTEL_LOGS_EXPORTER": "otlp" if telemetry.logs else "none",
        "OTEL_EXPORTER_OTLP_PROTOCOL": telemetry.protocol,
        "OTEL_METRIC_EXPORT_INTERVAL": str(telemetry.metrics_interval_ms),
        "OTEL_LOGS_EXPORT_INTERVAL": str(telemetry.logs_interval_ms),
        "OTEL_TRACES_EXPORT_INTERVAL": str(telemetry.traces_interval_ms),
    }
    if telemetry.traces:
        environment["OTEL_TRACES_EXPORTER"] = "otlp"
        environment["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] = "1"

    if telemetry.endpoint is not Missing:
        environment["OTEL_EXPORTER_OTLP_ENDPOINT"] = telemetry.endpoint

    if telemetry.headers is not Missing:
        environment["OTEL_EXPORTER_OTLP_HEADERS"] = telemetry.headers

    attributes: dict[str, str] = {
        "service.name": telemetry.service_name,
        **telemetry.resource_attributes,
        **(resource_attributes or {}),
    }
    environment["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(
        f"{quote(key, safe='')}={quote(value, safe='')}" for key, value in attributes.items()
    )

    if telemetry.log_prompts:
        environment["OTEL_LOG_USER_PROMPTS"] = "1"

    if telemetry.log_tool_details:
        environment["OTEL_LOG_TOOL_DETAILS"] = "1"

    if telemetry.log_tool_content:
        environment["OTEL_LOG_TOOL_CONTENT"] = "1"

    return environment


def build_launch(  # noqa: PLR0913
    config: ClaudeCodeConfig,
    /,
    *,
    binary: str,
    prompt: Sequence[Mapping[str, Any]],
    directory: str,
    home: str,
    config_directory: str,
    temporary_directory: str,
    instructions: str = "",
    session: ClaudeCodeSession = ClaudeCodeSession(),  # noqa: B008
    credential: str | None = None,
    credential_kind: Literal["api_key", "oauth_token"] = "api_key",
    base_url: str | None = None,
    trace_context: Mapping[str, str] | None = None,
    resource_attributes: Mapping[str, str] | None = None,
) -> ClaudeCodeLaunch:
    """Prepare the launch of Claude Code for a single prompt.

    Parameters
    ----------
    config
        Configuration of the run.
    binary
        Absolute path of the executable.
    prompt
        Content blocks of the user message.
    directory
        Working directory of the process.
    home
        Private home directory of the process.
    config_directory
        Directory of the Claude Code configuration.
    temporary_directory
        Directory of temporary files.
    instructions
        Instructions of the agent.
    session
        Persistence of the run within Claude Code sessions.
    credential
        API key or OAuth token.
    credential_kind
        Kind of the credential.
    base_url
        Custom API endpoint.
    trace_context
        Trace context of the executing agent.
    resource_attributes
        Resource attributes of the exported telemetry.

    Returns
    -------
    ClaudeCodeLaunch
        Launch ready to be run.

    Raises
    ------
    ValueError
        When the configuration is invalid.
    """
    validate_config(config)
    variable: str = (
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"
        if credential_kind == "oauth_token"
        else "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"
    )
    secrets: list[str] = [credential] if credential else []
    if (
        config.telemetry is not Missing
        and config.telemetry.headers is not Missing
        and config.telemetry.headers
    ):
        secrets.append(config.telemetry.headers)

    marker: str = uuid4().hex
    environment: Mapping[str, str] = claude_environment(
        config,
        home=home,
        config_directory=config_directory,
        temporary_directory=temporary_directory,
        base_url=base_url,
        trace_context=trace_context,
        resource_attributes=resource_attributes,
    )
    return ClaudeCodeLaunch(
        binary=binary,
        arguments=claude_arguments(config, instructions=instructions, session=session),
        environment={**environment, RUN_MARKER_VARIABLE: marker},
        directory=directory,
        prompt=json.dumps(
            {"type": "user", "message": {"role": "user", "content": list(prompt)}},
            separators=(",", ":"),
        ),
        credential=credential if credential else Missing,
        credential_variable=variable,
        marker=marker,
        secrets=tuple(secrets),
        timeout=config.timeout,
        grace=config.stop_grace,
    )
