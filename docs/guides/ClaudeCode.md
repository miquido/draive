# Claude Code agents

`draive.claude` runs [Claude Code](https://code.claude.com) as a draive `Agent`. Every call starts
the `claude` executable as a managed process, streams its work as agent output and stops it whatever
way the call ends. Nothing of the Claude Code SDK is imported - the protocol is spoken directly, so
there is no extra to install, only the `claude` executable itself. It requires a POSIX platform -
Linux, macOS, or WSL on Windows. Native Windows is not supported.

```python
from draive import ctx
from draive.claude import ClaudeCode, ClaudeCodeConfig, claude_code_agent

async with ctx.scope(
    "coding",
    ClaudeCodeConfig(
        model="sonnet",
        tools=("Read", "Grep", "Glob", "Edit"),
        directory="/path/to/project",  # the project, a temporary empty directory when not set
    ),
    disposables=(ClaudeCode(),),  # ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN, or credential=
):
    coder = claude_code_agent(
        "coder",
        description="Fixes bugs in the project",
        instructions="Fix the bug, change as little as possible.",
    )
    # the call is closed with the block - also when the loop is left early or fails
    async with ctx.closing(coder.call(input="The parser fails on empty input, fix it.")) as stream:
        async for chunk in stream:
            print(chunk)
```

## Authentication

Runs authenticate with the key or the token you give - never with the login of the user, unless you
give a `ClaudeCodeLocalLogin` (see
[Account connectors and local login](#account-connectors-and-local-login)). The credential is passed
through a file descriptor, so it is not part of the arguments or the environment of the process, and
the process has its own private home, so the sessions, keychain and settings of a logged in Claude
Code are out of its reach. When Claude Code reports to authenticate in any other way the run is
stopped, and entering the scope without a credential fails.

An OAuth token of a Team or Enterprise account may bring the `cc-plugin-sec-default` plugin its
organization enforces. No setting disables it, and it is not reported as a leak of the isolation.

| Source                                                                        | Meaning                                                          |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `ClaudeCode(credential=ClaudeCodeApiKey(...))`, `ANTHROPIC_API_KEY`           | Anthropic API key (`sk-ant-api...`)                              |
| `ClaudeCode(credential=ClaudeCodeOAuthToken(...))`, `CLAUDE_CODE_OAUTH_TOKEN` | OAuth token of `claude setup-token` (`sk-ant-oat...`)            |
| `ClaudeCode(credential=ClaudeCodeLocalLogin(...))`                            | local Claude Code login (`claude /login`), see below             |
| `ClaudeCode(base_url=...)`                                                    | custom API endpoint, i.e. a gateway - it can take any credential |

A credential is exactly one of the kinds - there is no pair of keys to choose between. Keys and
tokens, and the access token read from a login, can not be empty or larger than 4096 bytes, so they
fit the pipe they are handed over through.

The `claude` executable is resolved from `binary=`, `CLAUDE_CODE_BINARY`, the `PATH` and
`~/.local/bin/claude`.

## Account connectors and local login

The connectors of a claude.ai account (cloud MCP servers such as Slack or Drive, their tools are
named `mcp__claude_ai_<Name>__<tool>`) are off by default. `ClaudeCodeConfig(connectors=...)` turns
them on per run, so the same application can have them on for some runs and off for others:

- `ClaudeCodeNoConnectors()` - the default, only `mcp_servers` of the configuration are loaded.
- `ClaudeCodeAccountConnectors(wait=True)` - the connectors of the account are loaded in addition to
    the configured servers. With `wait` Claude Code waits (for a limited time, which it caps) for
    them to connect, otherwise the model does not see the tools of connectors which are not
    connected yet.

Nothing allows the tools of connectors for you - add them to `allowed_tools`, i.e.
`mcp__claude_ai_Slack__*`. Connectors the account never authenticated are reported by Claude Code as
`needs-auth` and are not counted as MCP servers which failed to connect, so they raise no warning.

Connectors need a token with the `user:mcp_servers` scope. API keys and `claude setup-token` tokens
never have it, so a run which asks for connectors with them fails with `ClaudeCodeUnavailable`
instead of silently seeing no tools. A `claude /login` login of a claude.ai account does have it,
`ClaudeCodeLocalLogin` uses it - a login without the scope fails with `ClaudeCodeUnavailable` when
connectors are requested:

```python
from draive import ctx
from draive.claude import (
    ClaudeCode,
    ClaudeCodeAccountConnectors,
    ClaudeCodeConfig,
    ClaudeCodeLocalLogin,
    claude_code_agent,
)

async with ctx.scope(
    "assistant",
    ClaudeCodeConfig(
        connectors=ClaudeCodeAccountConnectors(),
        allowed_tools=("mcp__claude_ai_Slack__*",),
    ),
    disposables=(
        ClaudeCode(credential=ClaudeCodeLocalLogin(config_directory="~/.claude")),
    ),
):
    assistant = claude_code_agent("assistant", instructions="Summarize my unread messages.")
    async with ctx.closing(assistant.call(input="What did I miss?")) as stream:
        async for chunk in stream:
            print(chunk)
```

`allowed_tools` replaces the default allowed set (`Read`, `Grep` and `Glob`), list those too if they
are wanted.

The login has to be made on this machine, as the user the process runs as. It is read from
`<config_directory>/.credentials.json` - only file based logins are supported, logins kept in the
keychain (macOS) are not, so there is usually no such file there, and entering the scope fails when
the file is missing. A file left over there is not updated by Claude Code, so on macOS entering the
scope also fails when its token has already expired. Every run reads the access token (valid for a
few hours) from it. When the token has less than `refresh_margin` seconds left (1800, 30 minutes, by
default), draive refreshes the login through Claude Code itself (`claude auth status`) under a lock
\- within the scope and across draive processes, as a refresh rotates the refresh token. The lock is
the file `<config_directory>/.login-refresh.lock`, so the directory has to be writable for
refreshes. The lock coordinates draive processes only, not the interactive Claude Code of the user,
which may refresh the same login. Use Claude Code 2.1.317 or newer for refreshes - older versions
may let `claude auth status` exit before a slow refresh is saved, which signs the login out; draive
does not check the version. A run gets the access token only - never the refresh token nor the
directory of the login - and still has its private home and configuration.

A refresh runs on its own task, a run which is cancelled or gives up on it does not stop it, and
exiting the `ClaudeCode` scope waits for the refreshes still in flight. A login has at most one
refresh in flight, shared by every run which needs it meanwhile. A run whose token is still valid
waits for a refresh at most `refresh_timeout` (15 seconds, and never past the expiry of the token)
and then goes on with its current token - also when the refresh fails, i.e. the lock file can not be
created or another process holds the lock longer than `lock_timeout` (30 seconds). A run whose token
has expired waits for the refresh to end, and fails with `ClaudeCodeUnavailable` only when it did
not help. `claude auth status` is killed only after `refresh_limit` (120 seconds), when the refresh
hangs.

Whether `claude auth status` refreshes at a margin as large as 30 minutes is not verified - Claude
Code may refresh only near the expiry. A refresh which does not extend the token, or which fails, is
logged, the token is used while it is valid, with a warning, and the refresh is retried after the
cooldown of a minute - a run whose token has already expired tries it again right away.

A run holds an access token and can not refresh it, so a run which outlives its token fails. A
refresh may also revoke the access tokens other processes still hold. Keep `refresh_margin` above
your longest run - `timeout` of the configuration is its upper bound, and a run may take
`stop_grace` longer. The default margin equals the default `timeout` (1800 seconds), so it is not
above the longest run unless raised. A token which is shorter than the margin but still valid is
used, with a warning.

### Filtering connectors

A run can use a part of the connectors only. `ClaudeCodeAccountConnectors` takes:

- `excluded` - display names as the account shows them (`"Slack"`, not `"claude.ai Slack"`), sent as
    `deniedMcpServers` entries named `claude.ai <name>` (Claude Code matches connectors by that name
    only),
- `excluded_urls` - URL patterns of connectors to remove, i.e. `https://mcp.atlassian.com/*`,
- `allowed` - `ClaudeCodeAllConnectors()` (default), or `ClaudeCodeConnectorsAt(urls=...)` with at
    least one URL pattern, the only connectors which are loaded.

```python
ClaudeCodeConfig(
    connectors=ClaudeCodeAccountConnectors(
        allowed=ClaudeCodeConnectorsAt(urls=("https://mcp.slack.com/*",)),
    ),
    mcp_servers={"docs": {"type": "http", "url": "https://docs.example"}},
)
```

An allowlist keeps only the connectors matching it and the servers of `mcp_servers`. It drops every
other MCP server - including the ones brought by plugins or subagents, which draive does not look
for. A connector can not be allowed by its name: an `allowedMcpServers` entry by `serverName` makes
Claude Code drop every connector. Hence the allowlist is by URL, and the servers of `mcp_servers`
are added to it by their names automatically, so a name has to be made of letters, digits, `_` and
`-` (the run fails with `ClaudeCodeUnavailable` otherwise). A list of URL patterns and names
together keeps both the matching connectors and the named servers - verified with Claude Code
2.1.331, while a list of URLs only drops the configured servers too. Claude Code prints a harmless
"blocked by enterprise policy" warning on its error output for every connector the filters remove.
Entries of `allowedMcpServers` and `deniedMcpServers` within `settings` are added to the derived
ones and have to be lists.

A connector which fails to connect or needs an authorization does not fail the run, and its tools
are missing. Both are recorded as the `claude_code.connectors_unavailable` attribute (naming at most
ten of them), and only the ones which failed raise a warning (`ClaudeCodeStarted.failed_connectors`)
\- most connectors which need an authorization were never authorized by the account, so a warning
about them would come with every run, while an expired grant of the account shows in the attribute.
Connectors removed by the filters are not reported by Claude Code at all, so they are not among
them.

## CLAUDE.md of the working directory

A run loads no `CLAUDE.md` by default - not the one of the working directory, nor the user's.
`setting_sources=("project",)` loads it together with the project settings, which can add hooks and
MCP servers. `claude_md=ClaudeCodeProjectClaudeMd()` loads the file without them:

```python
ClaudeCodeConfig(
    directory="/srv/project",
    claude_md=ClaudeCodeProjectClaudeMd(),
)
```

It reads `<directory>/CLAUDE.md` and `<directory>/.claude/CLAUDE.md` of the working directory. The
`directory` of the configuration is required - a temporary one holds no files, and a run without it
fails with `ClaudeCodeUnavailable`. It is done by passing the working directory as an additional one
with `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1`, which makes Claude Code read the files of
additional directories - nothing of `.claude/settings.json` is applied. It does not load the
`CLAUDE.md` of the directories above the working one, `CLAUDE.local.md`, the user's memory and files
in subdirectories (not even once the agent reads a file of one). The file permissions of the run are
not changed, the working directory is not added twice when it is among `additional_directories`.
Directories are compared as given and never resolved, so pass real paths - a symlink to the working
directory would be added again.

The same variable makes the files of every additional directory load, so they are excluded by
default (`additional_directories=ClaudeCodeSkipAdditionalClaudeMd()`) through `claudeMdExcludes` of
the settings, listing `CLAUDE.md` and `.claude/CLAUDE.md` of each by absolute path. How glob
characters within those paths are escaped is not verified, so a run with an additional directory
containing `*?[]{}` fails with `ClaudeCodeUnavailable` - its file would load. Whether
`.claude/rules/*.md` or `CLAUDE.local.md` of additional directories load with that variable as well
is not verified. Use `ClaudeCodeLoadAdditionalClaudeMd()` to load the files of additional
directories. Entries of `claudeMdExcludes` within `settings` are added to the derived ones (it has
to be a list).

## Isolation and permissions

A run is isolated by default and read-only. It has no user, project or local settings (and no
`CLAUDE.md` of the working directory, `claude_md` or `setting_sources=("project",)` opts in), no
auto memory, plugins or skills of the installation, no MCP servers beyond the configured ones
(unless account connectors are on), and only `Read`, `Grep` and `Glob` available. The environment is
built from scratch - only `PATH`, locale, proxy and certificate variables are inherited, anything
else of the parent process (especially the markers of a Claude Code running the parent) is left out.
Claude Code never asks for a permission: what is not allowed is denied.

`ClaudeCodeConfig` is a regular configuration state. It refuses what can not work when it is created
(`ValidationError`) - non-positive limits, relative directories, reserved environment variables - so
a run never starts with an invalid one:

| Field                                                        | Default                    |                                                                                                                             |
| ------------------------------------------------------------ | -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `model`, `effort`                                            | `sonnet`, model default    | model alias or name, reasoning effort                                                                                       |
| `permission_mode`                                            | `dontAsk`                  | `dontAsk`, `acceptEdits`, `plan`, `bypassPermissions`                                                                       |
| `tools`, `allowed_tools`, `disallowed_tools`                 | `Read`, `Grep`, `Glob`     | available tools, allowed ones (the available when not set, file tools within the directories of the run only), denied rules |
| `max_turns`, `max_budget_usd`                                | none                       | limits of a run                                                                                                             |
| `timeout`, `stop_grace`                                      | 30 min, 10 s               | the process is killed after the deadline, a stopped one has the grace to exit                                               |
| `directory`, `additional_directories`                        | temporary directory        | working directory (removed after the run when temporary) and more accessible directories                                    |
| `mcp_servers`, `subagents`, `plugin_directories`, `settings` | none                       | what Claude Code is given, nothing else is loaded (unless account connectors are on)                                        |
| `connectors`                                                 | `ClaudeCodeNoConnectors()` | account connectors, see [Account connectors and local login](#account-connectors-and-local-login)                           |
| `setting_sources`                                            | none                       | `project` loads the `CLAUDE.md` and `.claude` settings of the working directory                                             |
| `claude_md`                                                  | `ClaudeCodeNoClaudeMd`     | `ClaudeCodeProjectClaudeMd` loads the `CLAUDE.md` of the working directory without its settings                             |
| `env`, `inherit_env`                                         | none                       | extra variables, `CLAUDE*`, `ANTHROPIC_*`, `OTEL_*` and similar are reserved                                                |
| `stream_deltas`                                              | `False`                    | stream the text while it is generated                                                                                       |
| `system_prompt`                                              | `append`                   | whether instructions append to the system prompt (it keeps the tool guidance) or replace it                                 |

File tools are confined to the working and the additional directories by default - a symlink or `..`
leading out of them is refused, so the agent can not read the keys or the files of the user. `Bash`
is not confined to them, like any tool which is not about files. Use `bypassPermissions` and `Bash`
only within an environment which is safe to be modified - the agent has no sandbox other than its
own working directory and the tools you allow.

## Stopping runs

A run ends with its result. Everything else - the deadline, abandoning the call, cancelling the
task, a failure of the consumer - terminates the process (`SIGTERM`), which lets Claude Code stop
the commands of its tools, and kills it only when it does not exit within `stop_grace`. Its process
group is killed with it. The output is read until the result: children of the process which keep its
output open do not hold the call back, and the output of a process which exited is complete a second
later.

The call is an async generator, and the run is released when it is closed. Leaving the loop early,
or failing within it, does not close it - the process would run on until the generator is finalized.
Close it explicitly, `ctx.closing(...)` or `contextlib.aclosing(...)` do, as the example above does.
Cancelling the consuming task while it waits for the next chunk stops the run on its own.

Every run also marks the environment of the process and of everything it starts. On Linux, where the
environment of a process can be read exactly, whatever a tool left running with that environment - a
`nohup` or a `setsid` command of the agent - is stopped with the run. A command started with a
cleared environment is not found, and other platforms, i.e. macOS, have no reliable way of finding
the processes of a run: commands which the agent deliberately detaches from the run can outlive it
there. Run agents which have `Bash` within a container or a disposable environment.

## Output

The text of the agent is streamed as text parts. Everything else is reported as `ProcessingEvent`s:
`claude_code.tool_use` and `claude_code.tool_result` for each tool use, `claude_code.subagent_text`
for the subagents and `claude_code.result` at the end, with the cost, turns and duration. A failed
run raises a variant of `ClaudeCodeFailed`, while a run which can not start - authentication,
connectors, login resolution - raises `ClaudeCodeUnavailable` instead, and its variant
`ClaudeCodeLoginUnusable` when a `ClaudeCodeLocalLogin` gives no valid token - its credentials file
is missing, unreadable or malformed, or its token has expired and could not be refreshed - so
logging in again (`claude /login` as the user of the login) is what helps. Each variant of
`ClaudeCodeFailed` carries only what is known of its kind: `ClaudeCodeErrored` for an error the run
reported - an API failure, an exhausted limit - with its `failure` (`reason`, `api_error_status`,
`api_error_type`, `terminal_reason`) and its variant `ClaudeCodeAuthenticationFailed` when the API
rejected the credential (HTTP 401 or `authentication_error`) - a `ClaudeCodeLocalLogin` token which
was revoked or rotated by another process fails like that, and refreshing the login and running
again may help, `ClaudeCodeTimedOut`, `ClaudeCodeEnded` for a process which ended without a result
(`exit_code` and a redacted tail of the error output) and `ClaudeCodeOutputTooLarge`. All have the
`session_id` when the run reported one. Text and images are accepted as input.

## Context and memory

Claude Code manages its own context and compacts it when it grows. `compaction` tunes that:
`ClaudeCodeAutoCompaction(percent=, window=)` lowers the point of compaction,
`ClaudeCodeNoCompaction()` turns it off. What stays between calls is the `memory` of the agent, one
of:

- **Nothing (default).** Every call is a fresh session.

- **`AgentMemory` managed by draive.** The recalled history is handed to a fresh session and the
    turn is remembered afterwards, so any memory backend works. Claude Code does not see its earlier
    tool use, only the conversation.

    ```python
    coder = claude_code_agent("coder", memory=AgentMemory.volatile())
    ```

- **`ClaudeCodeSessions` managed by Claude Code.** The session of each thread is kept in the state
    directory and resumed by its next call, with the whole context Claude Code had - tool use,
    compaction and prompt caching included. The directories are stable, as sessions are stored by
    them, and the session of an interrupted run is resumed too.

    ```python
    coder = claude_code_agent("coder", memory=ClaudeCodeSessions(state_directory="/var/lib/agent"))
    ```

The memory is a single choice, so the two can not be combined. Injecting an arbitrary history into a
session, or reading how much of the context is used, is not possible with Claude Code.

## Telemetry

Claude Code exports OpenTelemetry - metrics, events and, as a beta, traces. Set `telemetry` to send
them to the collector used by the rest of the application. A run continues the trace of the agent
which called it, so its spans nest under the span of the agent.

```python
from draive.claude import ClaudeCodeConfig, ClaudeCodeTelemetry

ClaudeCodeConfig(
    telemetry=ClaudeCodeTelemetry.from_env(  # OTEL_EXPORTER_OTLP_* of the process
        signals={"metrics", "logs", "traces"},
    ),
)
```

`signals` choose what is exported - metrics and events by default, traces are a beta of Claude Code
and at least one is required. Prompts, tool details and tool content are not exported unless
`log_prompts`, `log_tool_details` and `log_tool_content` are set. Exports are short-lived like the
runs, so `export_interval_ms` is short by default. The result of each run is also recorded by draive
itself - the token usage as the usual `model.*` metrics (provider `claude_code`) and the cost as
`agent.claude_code.cost`.

## Testing

`ClaudeCodeRunner` is the installed capability, replace its `running` to fake the process (a runner
made this way resolves credentials on its own, `ClaudeCode` also waits for refreshes of a login):

```python
from draive.claude import ClaudeCodeResult, ClaudeCodeRunner, ClaudeCodeSuccess, ClaudeCodeText

async def running(launch):
    yield ClaudeCodeText(text="fixed")
    yield ClaudeCodeResult(session_id="s", outcome=ClaudeCodeSuccess(text="fixed"))

async with ctx.scope("test", ClaudeCodeRunner(binary="/fake", running=running)):
    ...
```
