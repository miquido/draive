# Claude Code agents

`draive.claude` runs [Claude Code](https://code.claude.com) as a draive `Agent`. Every call starts
the `claude` executable as a managed process, streams its work as agent output and stops it whatever
way the call ends. Nothing of the Claude Code SDK is imported - the protocol is spoken directly, so
there is no extra to install, only the `claude` executable itself.

```python
from draive import ctx
from draive.claude import ClaudeCode, ClaudeCodeConfig, claude_code_agent

async with ctx.scope(
    "coding",
    ClaudeCodeConfig(model="sonnet", tools=("Read", "Grep", "Glob", "Edit")),
    disposables=(ClaudeCode(),),  # ANTHROPIC_API_KEY, or oauth_token= / CLAUDE_CODE_OAUTH_TOKEN
):
    coder = claude_code_agent(
        "coder",
        description="Fixes bugs in the project",
        instructions="Fix the bug, change as little as possible.",
    )
    async for chunk in coder.call(input="The parser fails on empty input, fix it."):
        print(chunk)
```

## Authentication

Runs authenticate with the key or the token you give - never with the login of the user. The
credential is passed through a file descriptor, so it is not part of the arguments or the
environment of the process, and the process has its own private home, so the sessions, keychain and
settings of a logged in Claude Code are out of its reach. When Claude Code reports to authenticate
in any other way the run is stopped, and entering the scope without a credential fails.

An OAuth token of a Team or Enterprise account may bring the `cc-plugin-sec-default` plugin its
organization enforces. No setting disables it, and it is not reported as a leak of the isolation.

| Source                                                   | Meaning                                                          |
| -------------------------------------------------------- | ---------------------------------------------------------------- |
| `ClaudeCode(api_key=...)`, `ANTHROPIC_API_KEY`           | Anthropic API key (`sk-ant-api...`)                              |
| `ClaudeCode(oauth_token=...)`, `CLAUDE_CODE_OAUTH_TOKEN` | OAuth token of `claude setup-token` (`sk-ant-oat...`)            |
| `ClaudeCode(base_url=...)`                               | custom API endpoint, i.e. a gateway - it can take any credential |

The `claude` executable is resolved from `binary=`, `CLAUDE_CODE_BINARY`, the `PATH` and
`~/.local/bin/claude`.

## Isolation and permissions

A run is isolated by default and read-only. It has no user, project or local settings (and no
`CLAUDE.md` of the working directory, `setting_sources=("project",)` opts in), no auto memory,
plugins or skills of the installation, no MCP servers beyond the configured ones, and only `Read`,
`Grep` and `Glob` available. The environment is built from scratch - only `PATH`, locale, proxy and
certificate variables are inherited, anything else of the parent process (especially the markers of
a Claude Code running the parent) is left out. Claude Code never asks for a permission: what is not
allowed is denied.

`ClaudeCodeConfig` is a regular configuration state:

| Field                                                        | Default                 |                                                                                                                             |
| ------------------------------------------------------------ | ----------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `model`, `effort`                                            | `sonnet`, model default | model alias or name, reasoning effort                                                                                       |
| `permission_mode`                                            | `dontAsk`               | `dontAsk`, `acceptEdits`, `plan`, `bypassPermissions`                                                                       |
| `tools`, `allowed_tools`, `disallowed_tools`                 | `Read`, `Grep`, `Glob`  | available tools, allowed ones (the available when not set, file tools within the directories of the run only), denied rules |
| `max_turns`, `max_budget_usd`                                | none                    | limits of a run                                                                                                             |
| `timeout`, `stop_grace`                                      | 30 min, 10 s            | the process is killed after the deadline, a stopped one has the grace to exit                                               |
| `directory`, `additional_directories`                        | temporary directory     | working directory (removed after the run when temporary) and more accessible directories                                    |
| `mcp_servers`, `subagents`, `plugin_directories`, `settings` | none                    | what Claude Code is given, nothing else is loaded                                                                           |
| `setting_sources`                                            | none                    | `project` loads the `CLAUDE.md` and `.claude` settings of the working directory                                             |
| `env`, `inherit_env`                                         | none                    | extra variables, `CLAUDE*`, `ANTHROPIC_*`, `OTEL_*` and similar are reserved                                                |
| `stream_deltas`                                              | `False`                 | stream the text while it is generated                                                                                       |
| `system_prompt`                                              | `append`                | whether instructions append to the system prompt (it keeps the tool guidance) or replace it                                 |

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

Every run also marks the environment of the process and of everything it starts. On Linux, where the
environment of processes can be listed, whatever a tool left running - a `nohup` or a `setsid`
command of the agent - is stopped with the run. Some platforms, i.e. recent macOS, do not list it,
and commands which the agent deliberately detaches from the run can outlive it there. Run agents
which have `Bash` within a container or a disposable environment.

## Output

The text of the agent is streamed as text parts. Everything else is reported as `ProcessingEvent`s:
`claude_code.tool_use` and `claude_code.tool_result` for each tool use, `claude_code.subagent_text`
for the subagents and `claude_code.result` at the end, with the cost, turns and duration. A failed
run - an API failure, an exhausted limit, a timeout, a process which ended without a result - raises
`ClaudeCodeFailed` carrying the reason, the API status, the session and a redacted tail of the error
output. Text and images are accepted as input.

## Context and memory

Claude Code manages its own context and compacts it when it grows, `auto_compact`,
`auto_compact_percent` and `auto_compact_window` tune that. What stays between calls is up to you:

- **Nothing (default).** Every call is a fresh session.

- **`AgentMemory` managed by draive.** The recalled history is handed to a fresh session and the
    turn is remembered afterwards, so any memory backend works. Claude Code does not see its earlier
    tool use, only the conversation.

    ```python
    coder = claude_code_agent("coder", memory=AgentMemory.volatile())
    ```

- **Sessions managed by Claude Code.** With `persistence="resumable"` and a `state_directory` the
    session of each thread is kept and resumed by its next call, with the whole context Claude Code
    had - tool use, compaction and prompt caching included. The directories are stable, as sessions
    are stored by them. Claude Code sessions and `AgentMemory` are alternatives, using both is
    refused.

    ```python
    ClaudeCodeConfig(persistence="resumable", state_directory="/var/lib/agent")
    ```

Injecting an arbitrary history into a session, or reading how much of the context is used, is not
possible with Claude Code.

## Telemetry

Claude Code exports OpenTelemetry - metrics, events and, as a beta, traces. Set `telemetry` to send
them to the collector used by the rest of the application. A run continues the trace of the agent
which called it, so its spans nest under the span of the agent.

```python
from draive.claude import ClaudeCodeConfig, ClaudeCodeTelemetry

ClaudeCodeConfig(
    telemetry=ClaudeCodeTelemetry.from_env(traces=True),  # OTEL_EXPORTER_OTLP_* of the process
)
```

Prompts, tool details and tool content are not exported unless `log_prompts`, `log_tool_details` and
`log_tool_content` are set. Exports are short-lived like the runs, so the export intervals are short
by default. The result of each run is also recorded by draive itself - the token usage as the usual
`model.*` metrics (provider `claude_code`) and the cost as `agent.claude_code.cost`.

## Testing

`ClaudeCodeRunner` is the installed capability, replace its `running` to fake the process:

```python
from draive.claude import ClaudeCodeResult, ClaudeCodeRunner, ClaudeCodeText

async def running(launch):
    yield ClaudeCodeText(text="fixed")
    yield ClaudeCodeResult(session_id="s", text="fixed")

async with ctx.scope("test", ClaudeCodeRunner(binary="/fake", running=running)):
    ...
```
