import json
import os
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Final, cast, final
from urllib.parse import quote
from uuid import uuid4

from haiway import Missing, Sensitive, State, Verifier

from draive.claude.config import (
    ClaudeCodeAutoCompaction,
    ClaudeCodeConfig,
    ClaudeCodeNoCompaction,
    ClaudeCodeTelemetry,
)
from draive.claude.credentials import ClaudeCodeCredential
from draive.claude.sessions import (
    ClaudeCodeNewSession,
    ClaudeCodeNoSession,
    ClaudeCodeResumedSession,
    ClaudeCodeSession,
)
from draive.claude.workspace import ClaudeCodeWorkspace

__all__ = (
    "ClaudeCodeLaunch",
    "build_launch",
    "claude_arguments",
    "claude_environment",
    "claude_settings",
)

_NO_SESSION: Final[ClaudeCodeNoSession] = ClaudeCodeNoSession()

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

# variables of the environment which carry credentials, redacted from whatever is reported
_SENSITIVE_ENV: Final[Sequence[str]] = ("OTEL_EXPORTER_OTLP_HEADERS",)

# tools working with paths, allowed within the directories of the run only - their bare names would
# allow them to work with any path the process can reach, i.e. to read the keys of the user
_READ_TOOLS: Final[Sequence[str]] = ("Read", "Grep", "Glob")
# permission rules of every tool changing files are the ones of `Edit`, `Write(path)` never matches
_EDIT_TOOLS: Final[Sequence[str]] = ("Edit", "Write", "MultiEdit", "NotebookEdit")


def _marker(
    value: str,
) -> str:
    if not value:
        raise ValueError("A run needs a marker to be found by")

    return value


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
        Environment of the process without the credential descriptor and the marker, which the
        run adds. Values such as telemetry headers can be sensitive, it is never a part of the
        representation.
    directory
        Working directory of the process.
    prompt
        The ``user`` message sent to the process within a single ``stream-json`` line.
    credential
        The only credential of the run, handed over through a file descriptor.
    marker
        Value of ``DRAIVE_CLAUDE_RUN`` within the environment of the process and of everything it
        starts - also of what detaches or outlives it, so a run can stop all of it.
    timeout
        Limit of the run duration in seconds.
    grace
        Time a terminated process has to exit before it is killed, in seconds.
    """

    binary: str
    arguments: Sequence[str]
    environment: Annotated[Mapping[str, str], Sensitive()]
    directory: str
    prompt: str
    credential: ClaudeCodeCredential | Missing
    marker: Annotated[str, Verifier(_marker)]
    timeout: float
    grace: float

    @property
    def redactions(self) -> Sequence[str]:
        """Values redacted from anything reported out of the process output."""
        values: list[str] = [] if self.credential is Missing else [self.credential.value]
        values.extend(
            self.environment[name]
            for name in _SENSITIVE_ENV
            if self.environment.get(name)  # empty is not worth redacting
        )
        return tuple(values)


def claude_settings(
    config: ClaudeCodeConfig,
    /,
) -> str:
    """Prepare the isolating ``--settings`` JSON, configured settings are merged over it."""
    settings: dict[str, Any] = {
        "enabledPlugins": dict.fromkeys(_BUILTIN_PLUGINS, False),
        "autoMemoryEnabled": False,
    }
    if isinstance(config.compaction, ClaudeCodeNoCompaction):
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


def claude_arguments(
    config: ClaudeCodeConfig,
    /,
    *,
    instructions: str = "",
    session: ClaudeCodeSession = _NO_SESSION,
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
        *_plugin_arguments(config),
        *_permission_arguments(config),
        *_extension_arguments(config),
        *_session_arguments(session),
    ]
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


def _plugin_arguments(
    config: ClaudeCodeConfig,
    /,
) -> Sequence[str]:
    if not config.plugin_directories:
        return ("--disable-slash-commands",)  # plugins bring their own

    return tuple(
        argument for plugin in config.plugin_directories for argument in ("--plugin-dir", plugin)
    )


def _permission_arguments(
    config: ClaudeCodeConfig,
    /,
) -> Sequence[str]:
    allowed: Sequence[str] = (
        _scoped_tools(config) if config.allowed_tools is Missing else config.allowed_tools
    )
    arguments: list[str] = []
    if allowed:
        arguments.extend(("--allowedTools", ",".join(allowed)))

    if config.disallowed_tools:
        arguments.extend(("--disallowedTools", ",".join(config.disallowed_tools)))

    return tuple(arguments)


def _extension_arguments(
    config: ClaudeCodeConfig,
    /,
) -> Sequence[str]:
    arguments: list[str] = []
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
            ("--agents", json.dumps(config.subagents, separators=(",", ":"), default=_json_value))
        )

    for directory in config.additional_directories:
        arguments.extend(("--add-dir", directory))

    return tuple(arguments)


def _session_arguments(
    session: ClaudeCodeSession,
    /,
) -> Sequence[str]:
    match session:
        case ClaudeCodeNoSession():
            return ("--no-session-persistence",)

        case ClaudeCodeNewSession(identifier=identifier):
            return ("--session-id", identifier)

        case ClaudeCodeResumedSession(identifier=identifier):
            # a resumed session keeps the system prompt of its first request, which makes it
            # ignore instructions which changed since
            return ("--resume", identifier, "--system-prompt-snapshot", "off")


def _scoped_tools(
    config: ClaudeCodeConfig,
    /,
) -> Sequence[str]:
    rules: list[str] = []
    for tool in config.tools:
        if tool in _READ_TOOLS:
            scope: str = tool

        elif tool in _EDIT_TOOLS:
            scope = "Edit"

        else:
            rules.append(tool)
            continue

        for path in ("./**", *(_directory_pattern(path) for path in config.additional_directories)):
            if (rule := f"{scope}({path})") not in rules:
                rules.append(rule)

    return tuple(rules)


def _directory_pattern(
    directory: str,
    /,
) -> str:
    # rules are gitignore patterns, characters special within them would match other directories
    escaped: str = "".join(f"\\{char}" if char in "\\*?[]" else char for char in directory)
    # absolute paths are written with a double slash within the rules
    return f"//{escaped.lstrip('/')}/**"


def claude_environment(
    config: ClaudeCodeConfig,
    /,
    *,
    workspace: ClaudeCodeWorkspace,
    base_url: str | None = None,
    trace_context: Mapping[str, str] | None = None,
    resource_attributes: Mapping[str, str] | None = None,
    parent: Mapping[str, str] | None = None,
) -> Mapping[str, str]:
    """Prepare the environment of the process, nothing is taken from the parent besides a few
    connectivity variables.

    Parameters
    ----------
    config
        Configuration of the run.
    workspace
        Directories of the run, the process gets its private ones.
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
        Environment of the process, without the credential descriptor and the marker.
    """
    source: Mapping[str, str] = os.environ if parent is None else parent
    environment: dict[str, str] = {
        name: source[name] for name in (*_INHERITED_ENV, *config.inherit_env) if name in source
    }
    environment.update(
        {
            "HOME": workspace.home,
            "TMPDIR": workspace.temporary_directory,
            "CLAUDE_CONFIG_DIR": workspace.config_directory,
            "CLAUDE_CODE_TMPDIR": workspace.temporary_directory,
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
            "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
        }
    )
    if base_url:
        environment["ANTHROPIC_BASE_URL"] = base_url

    match config.compaction:
        case ClaudeCodeAutoCompaction(percent=percent, window=window):
            if percent is not Missing:
                environment["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(percent)

            if window is not Missing:
                environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(window)

        case ClaudeCodeNoCompaction():
            pass  # disabled by the settings

    if config.telemetry is not Missing:
        environment.update(
            _telemetry_environment(config.telemetry, trace_context, resource_attributes)
        )

    environment.update(config.env)
    return environment


def _telemetry_environment(
    telemetry: ClaudeCodeTelemetry,
    trace_context: Mapping[str, str] | None,
    resource_attributes: Mapping[str, str] | None,
    /,
) -> Mapping[str, str]:
    interval: str = str(telemetry.export_interval_ms)
    environment: dict[str, str] = {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_METRICS_EXPORTER": "otlp" if "metrics" in telemetry.signals else "none",
        "OTEL_LOGS_EXPORTER": "otlp" if "logs" in telemetry.signals else "none",
        "OTEL_EXPORTER_OTLP_PROTOCOL": telemetry.protocol,
        "OTEL_METRIC_EXPORT_INTERVAL": interval,
        "OTEL_LOGS_EXPORT_INTERVAL": interval,
        "OTEL_TRACES_EXPORT_INTERVAL": interval,
    }
    if "traces" in telemetry.signals:
        environment["OTEL_TRACES_EXPORTER"] = "otlp"
        environment["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] = "1"

    if telemetry.endpoint is not Missing:
        environment["OTEL_EXPORTER_OTLP_ENDPOINT"] = telemetry.endpoint

    if telemetry.headers is not Missing:
        environment["OTEL_EXPORTER_OTLP_HEADERS"] = telemetry.headers

    attributes: Mapping[str, str] = {
        "service.name": "claude-code",
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

    if telemetry.propagate_trace and trace_context:
        if traceparent := trace_context.get("traceparent"):
            environment["TRACEPARENT"] = traceparent

        if tracestate := trace_context.get("tracestate"):
            environment["TRACESTATE"] = tracestate

    return environment


def build_launch(
    config: ClaudeCodeConfig,
    /,
    *,
    binary: str,
    prompt: Sequence[Mapping[str, Any]],
    workspace: ClaudeCodeWorkspace,
    instructions: str = "",
    session: ClaudeCodeSession = _NO_SESSION,
    credential: ClaudeCodeCredential | Missing = Missing,
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
    workspace
        Directories of the run.
    instructions
        Instructions of the agent.
    session
        Persistence of the run within Claude Code sessions.
    credential
        The only credential of the run.
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
    """
    return ClaudeCodeLaunch(
        binary=binary,
        arguments=claude_arguments(config, instructions=instructions, session=session),
        environment=claude_environment(
            config,
            workspace=workspace,
            base_url=base_url,
            trace_context=trace_context,
            resource_attributes=resource_attributes,
        ),
        directory=workspace.directory,
        prompt=json.dumps(
            {"type": "user", "message": {"role": "user", "content": list(prompt)}},
            separators=(",", ":"),
        ),
        credential=credential,
        marker=uuid4().hex,
        timeout=config.timeout,
        grace=config.stop_grace,
    )
