import json
import os
import re
from collections.abc import Mapping, Sequence
from os.path import join, normpath
from typing import Annotated, Any, Final, Self, cast, final
from urllib.parse import quote
from uuid import uuid4

from haiway import BasicValue, Missing, Sensitive, State, Verifier

from draive.claude.config import (
    ClaudeCodeAccountConnectors,
    ClaudeCodeAllConnectors,
    ClaudeCodeAutoCompaction,
    ClaudeCodeConfig,
    ClaudeCodeConnectorsAt,
    ClaudeCodeLoadAdditionalClaudeMd,
    ClaudeCodeNoClaudeMd,
    ClaudeCodeNoCompaction,
    ClaudeCodeNoConnectors,
    ClaudeCodeNoSandbox,
    ClaudeCodeProjectClaudeMd,
    ClaudeCodeSandbox,
    ClaudeCodeSkipAdditionalClaudeMd,
    ClaudeCodeTelemetry,
)
from draive.claude.credentials import (
    ClaudeCodeAccessToken,
    ClaudeCodeApiKey,
    ClaudeCodeOAuthToken,
    ClaudeCodeResolvedCredential,
)
from draive.claude.exceptions import ClaudeCodeUnavailable
from draive.claude.sessions import (
    ClaudeCodeNewSession,
    ClaudeCodeNoSession,
    ClaudeCodeResumedSession,
    ClaudeCodeSession,
)
from draive.claude.tools import TOOLS_SERVER_KEY, TOOLS_SERVER_RULE
from draive.claude.workspace import ClaudeCodeWorkspace

__all__ = ("ClaudeCodeLaunch",)

_CONNECTORS_SCOPE: Final[str] = "user:mcp_servers"
# what an allowlist of Claude Code accepts as the name of a server
_ALLOWED_SERVER_NAME: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_-]+$")
# not escaped within `claudeMdExcludes`, as escaping is not verified
_GLOB_CHARACTERS: Final[str] = "*?[]{}"

_NO_SESSION: Final[ClaudeCodeNoSession] = ClaudeCodeNoSession()

# Claude Code registers these plugins on its own - each is disabled, a later version might add
# more which are reported by the started run
_BUILTIN_PLUGIN_PREFIX: Final[str] = "cc-plugin-"
_BUILTIN_PLUGINS: Final[Sequence[str]] = (
    "cc-plugin-agents-md",
    "cc-plugin-claude-platform",
    "cc-plugin-claude-test",
    "cc-plugin-diff",
    "cc-plugin-easel",
    "cc-plugin-mermaid",
    "cc-plugin-mods-guide",
    "cc-plugin-plugin-authoring",
    "cc-plugin-privacy",
    "cc-plugin-responsive-mode",
    "cc-plugin-sec-default",
    "cc-plugin-telemetry",
    "cc-plugin-tips",
    "cc-plugin-you-should-know",
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
    credential: ClaudeCodeResolvedCredential | Missing
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

    @classmethod
    def of(
        cls,
        config: ClaudeCodeConfig,
        /,
        *,
        binary: str,
        prompt: Sequence[Mapping[str, Any]],
        workspace: ClaudeCodeWorkspace,
        instructions: str = "",
        session: ClaudeCodeSession = _NO_SESSION,
        credential: ClaudeCodeResolvedCredential | Missing = Missing,
        base_url: str | None = None,
        trace_context: Mapping[str, str] | None = None,
        resource_attributes: Mapping[str, str] | None = None,
        tools_server: Mapping[str, BasicValue] | None = None,
    ) -> Self:
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
        tools_server
            MCP server entry of the draive tools, added under the reserved ``draive`` name and
            allowed.

        Returns
        -------
        ClaudeCodeLaunch
            Launch ready to be run.

        Raises
        ------
        ClaudeCodeUnavailable
            When account connectors are enabled and the credential can not use them, or the
            configuration of connectors or ``CLAUDE.md`` can not work.
        ValueError
            When ``tools_server`` is given and the configured MCP servers use the ``draive`` name.
        """
        cls.verify_connectors(config, credential)
        cls.verify_allowed_servers(config)
        cls.verify_claude_md(config)
        return cls(
            binary=binary,
            arguments=cls.prepare_arguments(
                config,
                instructions=instructions,
                session=session,
                directory=workspace.directory,
                tools_server=tools_server,
            ),
            environment=cls.prepare_environment(
                config,
                workspace=workspace,
                base_url=base_url,
                trace_context=trace_context,
                resource_attributes=resource_attributes,
                credential=credential,
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

    @staticmethod
    def leaked_plugins(
        config: ClaudeCodeConfig,
        plugins: Sequence[str],
        /,
    ) -> Sequence[str]:
        """Pick the built-in plugins a run loaded which its settings do not know.

        They come with a version of Claude Code newer than the isolating settings. Plugins
        which are disabled and loaded anyway are enforced by the organization of the
        account instead, see ``enforced_plugins``.

        Parameters
        ----------
        config
            Configuration of the run.
        plugins
            Names of the plugins the run reported.

        Returns
        -------
        Sequence[str]
            Names of the unknown built-in plugins.
        """
        enabled: Mapping[str, bool] = _enabled_plugins(config)
        return tuple(
            name
            for name in plugins
            if name.startswith(_BUILTIN_PLUGIN_PREFIX) and _builtin_key(name) not in enabled
        )

    @staticmethod
    def enforced_plugins(
        config: ClaudeCodeConfig,
        plugins: Sequence[str],
        /,
    ) -> Sequence[str]:
        """Pick the plugins a run loaded although its settings disable them.

        No setting disables a plugin which the organization of the account enforces. Plugins
        enabled by the settings are not among them.

        Parameters
        ----------
        config
            Configuration of the run.
        plugins
            Names of the plugins the run reported.

        Returns
        -------
        Sequence[str]
            Names of the disabled plugins which were loaded.
        """
        enabled: Mapping[str, bool] = _enabled_plugins(config)
        return tuple(name for name in plugins if enabled.get(_builtin_key(name)) is False)

    @staticmethod
    def prepare_settings(
        config: ClaudeCodeConfig,
        /,
        *,
        directory: str | None = None,
        tools_server: Mapping[str, BasicValue] | None = None,
    ) -> str:
        """Prepare the isolating ``--settings`` JSON, configured settings are merged over it.

        The lists ``claudeMdExcludes``, ``allowedMcpServers`` and ``deniedMcpServers`` derived
        from ``claude_md`` and ``connectors`` are not replaced by the configured settings, the
        entries of the configured ones are added to them instead. ``claudeMdExcludes`` always
        starts with the ``CLAUDE.md`` of the user. The ``sandbox`` derived from ``sandbox`` of
        the configuration, when ``Bash`` is among the tools, is merged with the configured one
        key by key, the configured keys win - except lists within its ``filesystem`` and
        ``network``, which the configured entries are added to.

        Parameters
        ----------
        config
            Configuration of the run.
        directory
            Working directory of the run, ``directory`` of the configuration when not given.
        tools_server
            MCP server entry of the draive tools, allowed next to the configured servers when
            connectors are allowed by an allowlist.

        Returns
        -------
        str
            The JSON of the settings.

        Raises
        ------
        ValueError
            When ``CLAUDE.md`` is loaded without a working directory.
        """
        settings: dict[str, Any] = {
            "autoMemoryEnabled": False,
        }
        if isinstance(config.compaction, ClaudeCodeNoCompaction):
            settings["autoCompactEnabled"] = False

        settings.update(config.settings)
        settings["enabledPlugins"] = _enabled_plugins(config)
        derived: dict[str, Sequence[Any]] = dict(_derived_settings(config, directory, tools_server))
        # project sources load the `CLAUDE.md` of every parent of the working directory, which
        # reaches the private one of the user within its home whenever the directory is under it,
        # configured exclusions extend the isolating ones instead of lifting them
        derived["claudeMdExcludes"] = (
            *_user_memory_patterns(),
            *derived.get("claudeMdExcludes", ()),
        )
        for key, value in derived.items():
            settings[key] = _merged_list(value, config.settings.get(key, Missing))

        sandbox: Mapping[str, Any] = _sandbox_settings(config)
        if sandbox:
            settings["sandbox"] = _merged_sandbox(
                sandbox,
                config.settings.get("sandbox", Missing),
            )

        return json.dumps(settings, separators=(",", ":"), default=_json_value)

    @staticmethod
    def prepare_arguments(
        config: ClaudeCodeConfig,
        /,
        *,
        instructions: str = "",
        session: ClaudeCodeSession = _NO_SESSION,
        directory: str | None = None,
        tools_server: Mapping[str, BasicValue] | None = None,
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
        directory
            Working directory of the run, ``directory`` of the configuration when not given. Needed
            only when the ``CLAUDE.md`` of the working directory is loaded.
        tools_server
            MCP server entry of the draive tools, added under the reserved ``draive`` name and
            allowed.

        Returns
        -------
        Sequence[str]
            Arguments of the executable. The prompt is not among them, it is sent to the input.

        Raises
        ------
        ValueError
            When ``tools_server`` is given and the configured MCP servers use the ``draive`` name,
            or ``CLAUDE.md`` is loaded without a working directory.
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
            ClaudeCodeLaunch.prepare_settings(
                config,
                directory=directory,
                tools_server=tools_server,
            ),
            *_connector_arguments(config),
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
            *_permission_arguments(config, tools_server),
            *_extension_arguments(config, directory, tools_server),
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

    @staticmethod
    def prepare_environment(
        config: ClaudeCodeConfig,
        /,
        *,
        workspace: ClaudeCodeWorkspace,
        base_url: str | None = None,
        trace_context: Mapping[str, str] | None = None,
        resource_attributes: Mapping[str, str] | None = None,
        parent: Mapping[str, str] | None = None,
        credential: ClaudeCodeResolvedCredential | Missing = Missing,
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
        credential
            The credential of the run, an access token declares its scopes within the environment.

        Returns
        -------
        Mapping[str, str]
            Environment of the process, without the credential descriptor and the marker.
        """
        environment: dict[str, str] = dict(
            ClaudeCodeLaunch.inherited_environment(parent, extra=config.inherit_env),
        )
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

        environment.update(_credential_environment(credential))
        environment.update(_connector_environment(config))
        environment.update(_claude_md_environment(config))

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

    @staticmethod
    def inherited_environment(
        parent: Mapping[str, str] | None = None,
        /,
        *,
        extra: Sequence[str] = (),
    ) -> Mapping[str, str]:
        """Pick the variables a process inherits - connectivity ones and the given extra names.

        Parameters
        ----------
        parent
            Environment inherited from, the one of the process by default.
        extra
            Names of additional variables to take.

        Returns
        -------
        Mapping[str, str]
            The inherited variables which are set.
        """
        source: Mapping[str, str] = os.environ if parent is None else parent
        return {name: source[name] for name in (*_INHERITED_ENV, *extra) if name in source}

    @staticmethod
    def verify_connectors(
        config: ClaudeCodeConfig,
        credential: ClaudeCodeResolvedCredential | Missing,
        /,
    ) -> None:
        """Refuse account connectors with a credential which can not use them.

        Parameters
        ----------
        config
            Configuration of the run.
        credential
            The resolved credential of the run.

        Raises
        ------
        ClaudeCodeUnavailable
            When connectors are enabled and the credential does not declare ``user:mcp_servers``.
        """
        match config.connectors:
            case ClaudeCodeNoConnectors():
                return

            case ClaudeCodeAccountConnectors():
                pass

        if credential is Missing:
            raise ClaudeCodeUnavailable("account connectors need a ClaudeCodeLocalLogin credential")

        match credential:
            case ClaudeCodeAccessToken(scopes=scopes) if _CONNECTORS_SCOPE in scopes:
                return

            case ClaudeCodeAccessToken():
                raise ClaudeCodeUnavailable(
                    f"the login has no '{_CONNECTORS_SCOPE}' scope, log in again with"
                    " `claude /login` on a claude.ai account"
                )

            case ClaudeCodeApiKey() | ClaudeCodeOAuthToken():
                raise ClaudeCodeUnavailable(
                    "account connectors need a ClaudeCodeLocalLogin, an API key or a token of"
                    f" `claude setup-token` has no '{_CONNECTORS_SCOPE}' scope"
                )

    @staticmethod
    def verify_claude_md(
        config: ClaudeCodeConfig,
        /,
    ) -> None:
        """Refuse loading ``CLAUDE.md`` which can not work with the rest of the configuration.

        Parameters
        ----------
        config
            Configuration of the run.

        Raises
        ------
        ClaudeCodeUnavailable
            When ``CLAUDE.md`` of the working directory is loaded without the ``directory`` of the
            configuration, or the additional directories are excluded and one has glob characters.
        """
        match config.claude_md:
            case ClaudeCodeNoClaudeMd():
                return

            case ClaudeCodeProjectClaudeMd(additional_directories=additional):
                if config.directory is Missing:
                    raise ClaudeCodeUnavailable(
                        "loading CLAUDE.md of the working directory needs the directory of the"
                        " configuration"
                    )

                match additional:
                    case ClaudeCodeLoadAdditionalClaudeMd():
                        return

                    case ClaudeCodeSkipAdditionalClaudeMd():
                        for other in config.additional_directories:
                            if any(char in other for char in _GLOB_CHARACTERS):
                                raise ClaudeCodeUnavailable(
                                    f"additional directory '{other}' has glob characters, its"
                                    " CLAUDE.md can not be excluded"
                                )

    @staticmethod
    def verify_allowed_servers(
        config: ClaudeCodeConfig,
        /,
    ) -> None:
        """Refuse an allowlist of connectors which can not allow the configured MCP servers.

        Parameters
        ----------
        config
            Configuration of the run.

        Raises
        ------
        ClaudeCodeUnavailable
            When connectors are allowed by URLs and a configured MCP server has a name which an
            allowlist can not match.
        """
        match config.connectors:
            case ClaudeCodeNoConnectors():
                return

            case ClaudeCodeAccountConnectors(allowed=allowed):
                match allowed:
                    case ClaudeCodeAllConnectors():
                        return

                    case ClaudeCodeConnectorsAt():
                        for name in config.mcp_servers:
                            if _ALLOWED_SERVER_NAME.fullmatch(name) is None:
                                raise ClaudeCodeUnavailable(
                                    f"MCP server '{name}' can not be allowed next to the allowed"
                                    " connectors, its name has to be made of letters, digits,"
                                    " '_' and '-'"
                                )


def _builtin_key(
    name: str,
    /,
) -> str:
    # plugins are reported without the marketplace their settings keys carry
    return f"{name}@builtin"


def _enabled_plugins(
    config: ClaudeCodeConfig,
    /,
) -> Mapping[str, bool]:
    # built-in plugins are disabled, configured ones are added - a plugin can be enabled explicitly
    return {
        **{_builtin_key(name): False for name in _BUILTIN_PLUGINS},
        **cast(Mapping[str, bool], config.settings.get("enabledPlugins", {})),
    }


def _user_memory_patterns() -> list[str]:
    home: str = os.path.realpath(os.path.expanduser("~"))
    # exclusions are matched as written - escaping a character keeps it from matching at all, and
    # a glob of a path with `(` or `{` matches nothing, an exact path matches still
    return [
        f"{home}/CLAUDE.md",
        f"{home}/CLAUDE.local.md",
        f"{home}/.claude/CLAUDE.md",
        f"{home}/.claude/**",  # its rules as well
    ]


def _escaped_pattern(
    path: str,
    /,
) -> str:
    # characters special within glob patterns would match other paths
    return "".join(f"\\{char}" if char in "\\*?[]" else char for char in path)


def _working_directory(
    config: ClaudeCodeConfig,
    directory: str | None,
    /,
) -> str:
    if directory is not None:
        return directory

    if config.directory is not Missing:
        return config.directory

    raise ValueError("Loading CLAUDE.md of the working directory needs the working directory")


def _derived_settings(
    config: ClaudeCodeConfig,
    directory: str | None,
    tools_server: Mapping[str, BasicValue] | None,
    /,
) -> Mapping[str, Sequence[Any]]:
    return {
        **_claude_md_settings(config, directory),
        **_connector_settings(config, tools_server),
    }


def _claude_md_settings(
    config: ClaudeCodeConfig,
    directory: str | None,
    /,
) -> Mapping[str, Sequence[Any]]:
    match config.claude_md:
        case ClaudeCodeNoClaudeMd():
            return {}

        case ClaudeCodeProjectClaudeMd(additional_directories=additional):
            match additional:
                case ClaudeCodeLoadAdditionalClaudeMd():
                    return {}

                case ClaudeCodeSkipAdditionalClaudeMd():
                    excluded: Sequence[str] = tuple(
                        path
                        for other in _other_directories(
                            config, _working_directory(config, directory)
                        )
                        for path in (join(other, "CLAUDE.md"), join(other, ".claude", "CLAUDE.md"))
                    )
                    return {"claudeMdExcludes": excluded} if excluded else {}


def _connector_settings(
    config: ClaudeCodeConfig,
    tools_server: Mapping[str, BasicValue] | None,
    /,
) -> Mapping[str, Sequence[Any]]:
    match config.connectors:
        case ClaudeCodeNoConnectors():
            return {}

        case ClaudeCodeAccountConnectors(
            excluded=excluded_names,
            excluded_urls=excluded_urls,
            allowed=allowed,
        ):
            derived: dict[str, Sequence[Any]] = {}
            denied: Sequence[Mapping[str, str]] = (
                *({"serverName": f"claude.ai {name}"} for name in excluded_names),
                *({"serverUrl": url} for url in excluded_urls),
            )
            if denied:
                derived["deniedMcpServers"] = denied

            match allowed:
                case ClaudeCodeAllConnectors():
                    pass

                case ClaudeCodeConnectorsAt(urls=urls):
                    # an allowlist drops every server it does not match, the configured ones and
                    # the draive tools too
                    names: Sequence[str] = (
                        (*config.mcp_servers, TOOLS_SERVER_KEY)
                        if tools_server is not None
                        else tuple(config.mcp_servers)
                    )
                    derived["allowedMcpServers"] = (
                        *({"serverUrl": url} for url in urls),
                        *({"serverName": name} for name in names),
                    )

            return derived


def _sandbox_settings(
    config: ClaudeCodeConfig,
    /,
) -> Mapping[str, Any]:
    if "Bash" not in config.tools:
        return {}

    match config.sandbox:
        case ClaudeCodeSandbox() as sandbox:
            # strict - nothing runs outside of the sandbox, nor without it
            settings: dict[str, Any] = {
                "enabled": True,
                "allowUnsandboxedCommands": False,
                "failIfUnavailable": True,
            }
            filesystem: dict[str, Any] = {
                key: tuple(paths)
                for key, paths in (
                    ("allowWrite", sandbox.allowed_writes),
                    ("denyWrite", sandbox.denied_writes),
                    ("denyRead", sandbox.denied_reads),
                    ("allowRead", sandbox.allowed_reads),
                )
                if paths
            }
            if filesystem:
                settings["filesystem"] = filesystem

            if sandbox.excluded_commands:
                settings["excludedCommands"] = tuple(sandbox.excluded_commands)

            network: dict[str, Any] = {
                key: tuple(domains)
                for key, domains in (
                    ("allowedDomains", sandbox.allowed_domains),
                    ("deniedDomains", sandbox.denied_domains),
                )
                if domains
            }
            if network:
                settings["network"] = network

            return settings

        case ClaudeCodeNoSandbox():
            return {}


def _merged_sandbox(
    derived: Mapping[str, Any],
    configured: object,
    /,
) -> Mapping[str, Any]:
    if configured is Missing:
        return derived

    # the configuration verifies it is a mapping
    configured_settings: Mapping[str, Any] = cast(Mapping[str, Any], configured)
    merged: dict[str, Any] = {**derived, **configured_settings}
    for key in ("filesystem", "network"):
        derived_section: object = derived.get(key, Missing)
        configured_section: object = configured_settings.get(key, Missing)
        if not isinstance(derived_section, Mapping) or not isinstance(configured_section, Mapping):
            continue  # nothing to merge, the configured one wins when present

        derived_keys: Mapping[str, Any] = cast(Mapping[str, Any], derived_section)
        configured_keys: Mapping[str, Any] = cast(Mapping[str, Any], configured_section)
        section: dict[str, Any] = {**derived_keys, **configured_keys}
        for name, value in derived_keys.items():
            added: object = configured_keys.get(name)
            # a configured list is added to the derived one, anything else replaces it
            if isinstance(added, Sequence) and not isinstance(added, str | bytes):
                section[name] = _merged_list(cast(Sequence[Any], value), configured_keys[name])

        merged[key] = section

    return merged


def _merged_list(
    derived: Sequence[Any],
    configured: object,
    /,
) -> Sequence[Any]:
    if configured is Missing:
        return derived

    # the configuration verifies it is a list
    return (*derived, *(item for item in cast(Sequence[Any], configured) if item not in derived))


def _other_directories(
    config: ClaudeCodeConfig,
    working: str,
    /,
) -> Sequence[str]:
    return tuple(
        dict.fromkeys(
            normpath(other)
            for other in config.additional_directories
            if normpath(other) != normpath(working)
        )
    )


def _json_value(
    value: object,
    /,
) -> object:
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, Any], value))

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return list(cast(Sequence[Any], value))

    raise TypeError(f"Unsupported setting value: {type(value).__name__}")


def _connector_arguments(
    config: ClaudeCodeConfig,
    /,
) -> Sequence[str]:
    match config.connectors:
        case ClaudeCodeNoConnectors():
            return ("--strict-mcp-config",)  # also keeps the connectors of the account out

        case ClaudeCodeAccountConnectors():
            return ()


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
    tools_server: Mapping[str, BasicValue] | None,
    /,
) -> Sequence[str]:
    allowed: Sequence[str] = (
        _scoped_tools(config) if config.allowed_tools is Missing else config.allowed_tools
    )
    if tools_server is not None:
        allowed = (*allowed, TOOLS_SERVER_RULE)

    arguments: list[str] = []
    if allowed:
        arguments.extend(("--allowedTools", ",".join(allowed)))

    if config.disallowed_tools:
        arguments.extend(("--disallowedTools", ",".join(config.disallowed_tools)))

    return tuple(arguments)


def _extension_arguments(
    config: ClaudeCodeConfig,
    directory: str | None,
    tools_server: Mapping[str, BasicValue] | None,
    /,
) -> Sequence[str]:
    servers: Mapping[str, Mapping[str, BasicValue]] = config.mcp_servers
    if tools_server is not None:
        if TOOLS_SERVER_KEY in servers:
            raise ValueError(
                f"The MCP server name '{TOOLS_SERVER_KEY}' is reserved for the draive tools"
            )

        servers = {**servers, TOOLS_SERVER_KEY: tools_server}

    arguments: list[str] = []
    if servers:
        arguments.extend(
            (
                "--mcp-config",
                json.dumps(
                    {"mcpServers": servers},
                    separators=(",", ":"),
                    default=_json_value,
                ),
            )
        )

    if config.subagents:
        arguments.extend(
            ("--agents", json.dumps(config.subagents, separators=(",", ":"), default=_json_value))
        )

    for additional in config.additional_directories:
        arguments.extend(("--add-dir", additional))

    match config.claude_md:
        case ClaudeCodeNoClaudeMd():
            pass

        case ClaudeCodeProjectClaudeMd():
            working: str = _working_directory(config, directory)
            # passed on its own, the permissions derived from the additional ones stay as they are
            if normpath(working) not in map(normpath, config.additional_directories):
                arguments.extend(("--add-dir", working))

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
    # rules are gitignore patterns, absolute paths are written with a double slash within them
    return f"//{_escaped_pattern(directory).lstrip('/')}/**"


def _credential_environment(
    credential: ClaudeCodeResolvedCredential | Missing,
    /,
) -> Mapping[str, str]:
    if credential is Missing:
        return {}

    match credential:
        case ClaudeCodeAccessToken(scopes=scopes):
            # a token handed over this way is assumed to have the inference scopes only, scopes
            # are not secret
            return {"CLAUDE_CODE_OAUTH_SCOPES": " ".join(scopes)}

        case ClaudeCodeApiKey() | ClaudeCodeOAuthToken():
            return {}


def _connector_environment(
    config: ClaudeCodeConfig,
    /,
) -> Mapping[str, str]:
    match config.connectors:
        case ClaudeCodeAccountConnectors(wait=wait):
            # connectors are not connected yet when the model starts otherwise
            return {"MCP_CONNECTION_NONBLOCKING": "0"} if wait else {}

        case ClaudeCodeNoConnectors():
            return {}


def _claude_md_environment(
    config: ClaudeCodeConfig,
    /,
) -> Mapping[str, str]:
    match config.claude_md:
        case ClaudeCodeProjectClaudeMd():
            return {"CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD": "1"}

        case ClaudeCodeNoClaudeMd():
            return {}


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
