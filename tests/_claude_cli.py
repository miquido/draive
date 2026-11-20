import os
import shlex
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from draive.claude import ClaudeCodeConfig, ClaudeCodeLaunch
from draive.claude.launch import ClaudeCodeSession, build_launch
from draive.claude.process import marked_processes

# a fake `claude` executable speaking the stream-json protocol, the scenario is selected by
# FAKE_CLAUDE_SCENARIO and whatever it received is written to FAKE_CLAUDE_RECORD
_SCRIPT = """\
#!{python}
import json, os, signal, subprocess, sys, time, uuid

scenario = os.environ.get("FAKE_CLAUDE_SCENARIO", "echo")
record = os.environ.get("FAKE_CLAUDE_RECORD")
line = sys.stdin.readline()
variable = next(
    (name for name in os.environ if name.endswith("_FILE_DESCRIPTOR")),
    None,
)
key = os.read(int(os.environ[variable]), 4096).decode() if variable else None
arguments = sys.argv[1:]


def after(flag):
    return arguments[arguments.index(flag) + 1] if flag in arguments else None


session = after("--session-id") or after("--resume") or str(uuid.uuid4())
if "--no-session-persistence" not in arguments and os.environ.get("CLAUDE_CONFIG_DIR"):
    # the transcript is written while the session runs, a finished one or not
    project = os.path.join(os.environ["CLAUDE_CONFIG_DIR"], "projects", "fake-project")
    os.makedirs(project, exist_ok=True)
    open(os.path.join(project, session + ".jsonl"), "a").close()
prompt = json.loads(line)["message"]["content"] if line.strip() else []
text = "\\n".join(block["text"] for block in prompt if block.get("type") == "text")
if record:
    with open(record, "w") as file:
        json.dump(
            {{
                "arguments": arguments,
                "environment": dict(os.environ),
                "line": line,
                "key": key,
                "variable": variable,
                "directory": os.getcwd(),
                "text": text,
            }},
            file,
        )


def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()


def result(**fields):
    emit(
        {{
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": session,
            "result": fields.pop("result", "ok"),
            "num_turns": 1,
            "duration_ms": 12,
            "total_cost_usd": 0.5,
            "modelUsage": {{
                "fake-model": {{
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadInputTokens": 2,
                    "cacheCreationInputTokens": 1,
                    "costUSD": 0.5,
                }}
            }},
            "permission_denials": [],
            **fields,
        }}
    )


def assistant(*blocks, **fields):
    message = {{"role": "assistant", "content": list(blocks)}}
    emit({{"type": "assistant", "message": message, **fields}})


if scenario in ("hang", "orphans"):  # children of the process, to be stopped with it
    detached = bool(os.environ.get("FAKE_CLAUDE_DETACH"))
    children = [subprocess.Popen(["sleep", "300"], start_new_session=detached)]
    if scenario == "orphans":  # one in the group and one which left it, both outlive the process
        children.append(subprocess.Popen(["sleep", "300"], start_new_session=True))
    with open(os.environ["FAKE_CLAUDE_CHILD"], "w") as file:
        file.write(" ".join(str(child.pid) for child in children))

emit(
    {{
        "type": "system",
        "subtype": "init",
        "session_id": session,
        "model": "fake-model",
        "claude_code_version": "0.0.0",
        "permissionMode": "dontAsk",
        "apiKeySource": os.environ.get("FAKE_CLAUDE_SOURCE")
        or ("ANTHROPIC_API_KEY" if (variable or "").startswith("CLAUDE_CODE_API_KEY") else "none"),
        "tools": [],
        "mcp_servers": json.loads(os.environ.get("FAKE_CLAUDE_MCP", "[]")),
        "plugins": {plugins},
    }}
)

if scenario == "echo":
    assistant({{"type": "text", "text": "echo: " + text}})
    result(result="echo: " + text)

elif scenario == "tools":
    assistant(
        {{"type": "text", "text": "reading"}},
        {{"type": "tool_use", "id": "t1", "name": "Read", "input": {{"path": "a.txt"}}}},
    )
    emit(
        {{
            "type": "user",
            "message": {{
                "role": "user",
                "content": [{{"type": "tool_result", "tool_use_id": "t1", "content": "file body"}}],
            }},
        }}
    )
    assistant({{"type": "text", "text": "done"}})
    result(result="done")

elif scenario == "deltas":
    for part in ("hel", "lo"):
        emit(
            {{
                "type": "stream_event",
                "event": {{
                    "type": "content_block_delta",
                    "delta": {{"type": "text_delta", "text": part}},
                }},
            }}
        )
    assistant({{"type": "text", "text": "hello"}})
    result(result="hello")

elif scenario == "error401":
    assistant(
        {{"type": "text", "text": "Invalid API key"}},
        error="authentication_failed",
        is_api_error_message=True,
    )
    result(
        result="Invalid API key",
        is_error=True,
        api_error_status=401,
        terminal_reason="api_error",
    )
    sys.exit(1)

elif scenario == "bigline":
    emit(
        {{
            "type": "user",
            "message": {{
                "role": "user",
                "content": [
                    {{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 300000}}
                ],
            }},
        }}
    )
    result(result="big")

elif scenario == "stderr_flood":
    sys.stderr.write("e" * 3000000)
    sys.stderr.flush()
    assistant({{"type": "text", "text": "survived"}})
    result(result="survived")

elif scenario == "garbage":
    sys.stdout.write("not json at all\\n[1, 2]\\n{{}}\\n")
    sys.stdout.flush()
    assistant({{"type": "text", "text": "after garbage"}})
    result(result="after garbage")

elif scenario == "init_then_exit":
    sys.stderr.write("crashed after init" + chr(10))
    sys.exit(4)

elif scenario == "split_secret":
    sys.stderr.write("E" * 9000 + key[:10])
    sys.stderr.flush()
    time.sleep(0.4)
    sys.stderr.write(key[10:] + chr(10))
    sys.stderr.flush()
    sys.exit(3)

elif scenario == "late_output":
    code = (
        "import sys, time; time.sleep(2.5); "
        "sys.stdout.write('late'); sys.stdout.flush(); "
        "sys.stderr.write('late'); sys.stderr.flush()"
    )
    child = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    with open(os.environ["FAKE_CLAUDE_CHILD"], "w") as file:
        file.write(str(child.pid))
    assistant({{"type": "text", "text": "done"}})
    result(result="done")

elif scenario == "noresult":
    sys.stderr.write("fatal: bad credentials " + (key or "") + "\\n")
    sys.exit(3)

elif scenario == "orphans":
    assistant({{"type": "text", "text": "leaving children behind"}})
    result(result="done")

elif scenario == "hang":
    if os.environ.get("FAKE_CLAUDE_STUBBORN"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    elif os.environ.get("FAKE_CLAUDE_GRACEFUL"):
        def terminated(*_):
            with open(os.environ["FAKE_CLAUDE_GRACEFUL"], "w") as file:
                file.write("terminated")
            sys.exit(0)
        signal.signal(signal.SIGTERM, terminated)
    assistant({{"type": "text", "text": "working"}})
    while True:
        time.sleep(0.1)
"""


def write_fake_claude(
    directory: Path,
    /,
    *,
    plugins: str = "[]",
) -> str:
    """Write the fake executable, returns its path."""
    script: Path = directory / "claude.py"
    script.write_text(_SCRIPT.format(python="/usr/bin/env python3", plugins=plugins))
    # a shebang can not take an interpreter path containing spaces, the launcher can
    path: Path = directory / "claude"
    command: str = f"exec {shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    path.write_text(f'#!/bin/sh\n{command} "$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def launch_for(
    binary: str,
    directory: Path,
    /,
    scenario: str = "echo",
    *,
    prompt: str = "hello",
    credential: str | None = "sk-ant-api03-secret-credential",
    session: ClaudeCodeSession = ClaudeCodeSession(),  # noqa: B008
    timeout: float = 30.0,
    grace: float = 2.0,
    env: Mapping[str, str] | None = None,
    **config: object,
) -> ClaudeCodeLaunch:
    """Prepare a launch of the fake executable within the given directory."""
    (directory / "work").mkdir(exist_ok=True)
    return build_launch(
        ClaudeCodeConfig(
            timeout=timeout,
            stop_grace=grace,
            env={
                "FAKE_CLAUDE_SCENARIO": scenario,
                "FAKE_CLAUDE_RECORD": str(directory / "record.json"),
                "FAKE_CLAUDE_CHILD": str(directory / "child.pid"),
                **(env or {}),
            },
            **config,  # pyright: ignore[reportArgumentType]
        ),
        binary=binary,
        prompt=({"type": "text", "text": prompt},),
        directory=str(directory / "work"),
        home=str(directory),
        config_directory=str(directory),
        temporary_directory=str(directory),
        session=session,
        credential=credential,
    )


def environment_listing_available() -> bool:
    """Whether processes can be found by the marker in their environment on this platform."""
    marker = uuid4().hex
    child = subprocess.Popen(
        ["sleep", "30"],
        env={**os.environ, "DRAIVE_CLAUDE_RUN": marker},
    )
    try:
        time.sleep(0.3)
        return child.pid in marked_processes(marker)

    finally:
        child.kill()
        child.wait()


def alive(
    pid: int,
    /,
) -> bool:
    """Whether the process is running."""
    try:
        os.kill(pid, 0)

    except ProcessLookupError:
        return False

    # a stopped child which nobody reaped - i.e. within a container - still has its identifier
    state: str = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")
