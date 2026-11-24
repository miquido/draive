import asyncio
import json
import os
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pytest
from haiway import ctx

from draive.claude import (
    ClaudeCode,
    ClaudeCodeAccessToken,
    ClaudeCodeAccountConnectors,
    ClaudeCodeApiKey,
    ClaudeCodeConfig,
    ClaudeCodeLocalLogin,
    ClaudeCodeUnavailable,
    claude_code_agent,
)
from draive.claude.login import ClaudeCodeLogins
from tests._claude_cli import alive, write_fake_claude

TOKEN = "sk-ant-oat01-login-access-token"
REFRESHED = "sk-ant-oat01-login-refreshed-token"
REFRESH_TOKEN = "sk-ant-ort01-never-in-a-run"
HOUR_MS = 3_600_000

# fake `claude auth status` - records its invocations and rewrites the credentials file the way a
# refresh does, the `mode` file of the login directory selects the outcome (the environment of
# the refresh is not inherited)
_FAKE_BINARY = """\
#!{python}
import json, os, sys, time

directory = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.environ["HOME"], ".claude")
with open(os.path.join(directory, "invocations"), "a") as file:
    file.write(
        " ".join(sys.argv[1:])
        + "|" + os.environ["HOME"]
        + "|" + os.environ.get("CLAUDE_CONFIG_DIR", "-")
        + "\\n"
    )
mode_file = os.path.join(directory, "mode")
mode = open(mode_file).read() if os.path.exists(mode_file) else "refresh"
if mode == "hang":
    with open(os.path.join(directory, "pid"), "w") as file:
        file.write(str(os.getpid()))
    time.sleep(30)
time.sleep(0.3)
if mode == "refresh":
    with open(os.path.join(directory, ".credentials.json"), "w") as file:
        json.dump(
            {{
                "claudeAiOauth": {{
                    "accessToken": "{refreshed}",
                    "refreshToken": "{refresh}",
                    "expiresAt": int(time.time() * 1000) + 3 * {hour},
                    "scopes": ["user:inference", "user:mcp_servers"],
                }}
            }},
            file,
        )
"""

# holds the lock of a login refresh for `seconds`, optionally replacing the credentials like a
# refresh of another process does
_HOLDER = """\
import fcntl, json, os, sys, time

directory, seconds, refreshed = sys.argv[1], float(sys.argv[2]), sys.argv[3]
descriptor = os.open(os.path.join(directory, ".login-refresh.lock"), os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX)
print("held", flush=True)
time.sleep(seconds)
if refreshed:
    with open(os.path.join(directory, ".credentials.json"), "w") as file:
        json.dump(
            {{
                "claudeAiOauth": {{
                    "accessToken": refreshed,
                    "refreshToken": "{refresh}",
                    "expiresAt": int(time.time() * 1000) + 3 * {hour},
                    "scopes": ["user:inference", "user:mcp_servers"],
                }}
            }},
            file,
        )
"""

_PROBE = """\
import fcntl, os, sys

descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
try:
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(1)
"""


def _login_directory(
    root: Path,
    *,
    name: str = "login",
    expires_in_ms: int = 3 * HOUR_MS,
    scopes: Any = ("user:inference", "user:mcp_servers"),
) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": TOKEN,
                    "refreshToken": REFRESH_TOKEN,
                    "expiresAt": int(time.time() * 1000) + expires_in_ms,
                    "scopes": list(scopes),
                }
            }
        )
    )
    return directory


def _fake_binary(root: Path) -> str:
    path = root / "claude-refresh"
    path.write_text(
        _FAKE_BINARY.format(
            python=sys.executable,
            refreshed=REFRESHED,
            refresh=REFRESH_TOKEN,
            hour=HOUR_MS,
        )
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _invocations(directory: Path) -> list[str]:
    file = directory / "invocations"
    return file.read_text().splitlines() if file.exists() else []


def _hold_lock(
    directory: Path,
    seconds: float,
    refreshed: str = "",
) -> subprocess.Popen[str]:
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _HOLDER.format(refresh=REFRESH_TOKEN, hour=HOUR_MS),
            str(directory),
            str(seconds),
            refreshed,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "held"
    return holder


def _lock_is_free(directory: Path) -> bool:
    return (
        subprocess.run(
            [sys.executable, "-c", _PROBE, str(directory / ".login-refresh.lock")],
            check=False,
        ).returncode
        == 0
    )


async def _until(condition: Any, seconds: float = 10) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not met"
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_a_fresh_token_is_read_as_it_is(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path)

    token = await ClaudeCodeLogins().resolve(
        ClaudeCodeLocalLogin(config_directory=str(directory)),
        binary=_fake_binary(tmp_path),
    )

    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == TOKEN
    assert tuple(token.scopes) == ("user:inference", "user:mcp_servers")
    assert TOKEN not in repr(token)
    assert _invocations(directory) == []


@pytest.mark.asyncio
async def test_a_token_about_to_expire_is_refreshed_through_claude_code(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=60_000)

    async with ctx.scope("test"):
        token = await ClaudeCodeLogins().resolve(
            ClaudeCodeLocalLogin(config_directory=str(directory)),
            binary=_fake_binary(tmp_path),
        )

    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == REFRESHED
    assert REFRESH_TOKEN not in repr(token)
    arguments, home, config = _invocations(directory)[0].split("|")
    assert arguments == "auth status"
    assert home == str(Path.home())
    assert config == str(directory)  # not the default directory, so it is passed


@pytest.mark.asyncio
async def test_the_default_directory_is_not_passed_to_the_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    directory = _login_directory(home, name=".claude", expires_in_ms=60_000)
    monkeypatch.setenv("HOME", str(home))

    async with ctx.scope("test"):
        token = await ClaudeCodeLogins().resolve(
            ClaudeCodeLocalLogin(), binary=_fake_binary(tmp_path)
        )

    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == REFRESHED
    _, seen_home, config = _invocations(directory)[0].split("|")
    assert seen_home == str(home)
    assert config == "-"


@pytest.mark.asyncio
async def test_a_refresh_which_does_not_help_fails_when_expired(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=-60_000)
    (directory / "mode").write_text("nothing")

    async with ctx.scope("test"):
        with pytest.raises(ClaudeCodeUnavailable, match="expired"):
            await ClaudeCodeLogins().resolve(
                ClaudeCodeLocalLogin(config_directory=str(directory)),
                binary=_fake_binary(tmp_path),
            )

    assert len(_invocations(directory)) == 1


@pytest.mark.asyncio
async def test_a_valid_token_shorter_than_the_margin_is_used_without_another_attempt(
    tmp_path: Path,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)
    (directory / "mode").write_text("nothing")
    login = ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200)
    logins = ClaudeCodeLogins()
    binary = _fake_binary(tmp_path)

    async with ctx.scope("test"):
        tokens = [await logins.resolve(login, binary=binary) for _ in range(3)]

    assert [token.value for token in tokens if isinstance(token, ClaudeCodeAccessToken)] == [
        TOKEN
    ] * 3
    assert len(_invocations(directory)) == 1  # the cooldown


@pytest.mark.asyncio
async def test_a_refresh_is_attempted_again_after_the_cooldown(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)
    (directory / "mode").write_text("nothing")
    login = ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200)
    logins = ClaudeCodeLogins(cooldown=0)
    binary = _fake_binary(tmp_path)

    async with ctx.scope("test"):
        for _ in range(2):
            await logins.resolve(login, binary=binary)

    assert len(_invocations(directory)) == 2


@pytest.mark.asyncio
async def test_a_refresh_which_hangs_is_killed(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)
    (directory / "mode").write_text("hang")

    async with ctx.scope("test"):
        token = await ClaudeCodeLogins(refresh_timeout=0.5).resolve(
            ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200),
            binary=_fake_binary(tmp_path),
        )

    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == TOKEN  # still valid
    assert not alive(int((directory / "pid").read_text()))


@pytest.mark.asyncio
async def test_a_refresh_which_can_not_start_is_survived_while_the_token_is_valid(
    tmp_path: Path,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)

    async with ctx.scope("test"):
        token = await ClaudeCodeLogins().resolve(
            ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200),
            binary=str(tmp_path / "missing"),
        )

    assert isinstance(token, ClaudeCodeAccessToken)


@pytest.mark.asyncio
async def test_concurrent_resolutions_refresh_once(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=60_000)
    binary = _fake_binary(tmp_path)
    login = ClaudeCodeLocalLogin(config_directory=str(directory))
    logins = ClaudeCodeLogins()

    async with ctx.scope("test"):
        tokens = await asyncio.gather(*(logins.resolve(login, binary=binary) for _ in range(5)))

    assert {token.value for token in tokens if isinstance(token, ClaudeCodeAccessToken)} == {
        REFRESHED
    }
    assert len(_invocations(directory)) == 1


def test_resolutions_refresh_once_within_different_event_loops(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=60_000)
    binary = _fake_binary(tmp_path)
    login = ClaudeCodeLocalLogin(config_directory=str(directory))

    async def resolving() -> None:
        async with ctx.scope("test"):
            await ClaudeCodeLogins().resolve(login, binary=binary)

    asyncio.run(resolving())
    asyncio.run(resolving())

    assert len(_invocations(directory)) == 1


@pytest.mark.asyncio
async def test_a_lock_of_another_process_is_waited_for_and_its_refresh_used(
    tmp_path: Path,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=60_000)
    holder = _hold_lock(directory, 0.5, refreshed=REFRESHED)

    async with ctx.scope("test"):
        token = await ClaudeCodeLogins().resolve(
            ClaudeCodeLocalLogin(config_directory=str(directory)),
            binary=_fake_binary(tmp_path),
        )

    holder.wait()
    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == REFRESHED
    assert _invocations(directory) == []  # read again under the lock, nothing left to refresh


@pytest.mark.asyncio
async def test_a_lock_held_for_too_long_fails(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=60_000)
    holder = _hold_lock(directory, 3)

    try:
        async with ctx.scope("test"):
            with pytest.raises(ClaudeCodeUnavailable, match="for too long"):
                await ClaudeCodeLogins(lock_timeout=0.3).resolve(
                    ClaudeCodeLocalLogin(config_directory=str(directory)),
                    binary=_fake_binary(tmp_path),
                )

    finally:
        holder.kill()
        holder.wait()

    assert _invocations(directory) == []


@pytest.mark.asyncio
async def test_a_cancelled_refresh_is_completed_and_releases_everything(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=60_000)
    binary = _fake_binary(tmp_path)
    login = ClaudeCodeLocalLogin(config_directory=str(directory))
    logins = ClaudeCodeLogins()
    descriptors = len(os.listdir("/dev/fd"))

    async with ctx.scope("test"):
        task = asyncio.ensure_future(logins.resolve(login, binary=binary))
        await _until(lambda: bool(_invocations(directory)))  # the refresh is running
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # not left half done - the new token is written
        await _until(lambda: REFRESHED in (directory / ".credentials.json").read_text())
        await _until(lambda: _lock_is_free(directory))
        token = await logins.resolve(login, binary=binary)

    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == REFRESHED
    assert len(_invocations(directory)) == 1
    assert len(os.listdir("/dev/fd")) <= descriptors


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [0.001, 0.05, 0.2])
async def test_a_cancelled_wait_for_the_lock_never_leaks_it(tmp_path: Path, delay: float) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=60_000)
    binary = _fake_binary(tmp_path)
    login = ClaudeCodeLocalLogin(config_directory=str(directory))
    logins = ClaudeCodeLogins()
    holder = _hold_lock(directory, 0.4)
    descriptors = len(os.listdir("/dev/fd"))

    async with ctx.scope("test"):
        task = asyncio.ensure_future(logins.resolve(login, binary=binary))
        await asyncio.sleep(delay)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        holder.wait()
        # whatever the cancelled resolution left running finishes and lets go of the lock
        await _until(lambda: _lock_is_free(directory))
        token = await ClaudeCodeLogins(lock_timeout=1).resolve(login, binary=binary)

    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == REFRESHED
    assert len(os.listdir("/dev/fd")) <= descriptors


@pytest.mark.asyncio
async def test_a_missing_credentials_file_is_reported(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    login = ClaudeCodeLocalLogin(config_directory=str(tmp_path / "empty"))

    with pytest.raises(ClaudeCodeUnavailable, match=r"claude /login"):
        await ClaudeCodeLogins().resolve(login, binary="/bin/claude")


def _chain(exception: BaseException | None) -> list[BaseException]:
    chain: list[BaseException] = []
    while exception is not None and exception not in chain:
        chain.append(exception)
        chain.extend(_chain(exception.__cause__))
        exception = exception.__context__

    return chain


def _holds(exception: BaseException, codeword: str) -> bool:
    for current in _chain(exception):
        if codeword in str(current) or codeword in repr(current) or codeword in repr(vars(current)):
            return True

        for frame, _ in traceback.walk_tb(current.__traceback__):
            # the frames of this test hold the codeword they search for
            if frame.f_code.co_filename != __file__ and codeword in repr(frame.f_locals):
                return True

    return False


_MALFORMED = {
    "not json": ("not json " + REFRESH_TOKEN).encode(),
    "not utf-8": b"\xff\xfe" + REFRESH_TOKEN.encode(),
    "array": json.dumps([REFRESH_TOKEN]).encode(),
    "string": json.dumps({"claudeAiOauth": REFRESH_TOKEN}).encode(),
    "no token": json.dumps({"claudeAiOauth": {"refreshToken": REFRESH_TOKEN}}).encode(),
    "bad expiry": json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": TOKEN,
                "refreshToken": REFRESH_TOKEN,
                "expiresAt": "soon",
                "scopes": [],
            }
        }
    ).encode(),
    "empty token": json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": "",
                "refreshToken": REFRESH_TOKEN,
                "expiresAt": 1,
                "scopes": [],
            }
        }
    ).encode(),
    "bad scopes": json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": TOKEN,
                "refreshToken": REFRESH_TOKEN,
                "expiresAt": 1,
                "scopes": [1],
            }
        }
    ).encode(),
    "no scopes": json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": TOKEN,
                "refreshToken": REFRESH_TOKEN,
                "expiresAt": 1,
            }
        }
    ).encode(),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", list(_MALFORMED))
async def test_a_malformed_credentials_file_does_not_leak(tmp_path: Path, variant: str) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / ".credentials.json").write_bytes(_MALFORMED[variant])
    login = ClaudeCodeLocalLogin(config_directory=str(directory))

    with pytest.raises(ClaudeCodeUnavailable, match="malformed") as exc_info:
        await ClaudeCodeLogins().resolve(login, binary="/bin/claude")

    exception = exc_info.value
    assert exception.__cause__ is None
    assert exception.__context__ is None
    for codeword in (REFRESH_TOKEN, TOKEN):
        assert not _holds(exception, codeword)


@pytest.mark.asyncio
async def test_static_credentials_are_resolved_to_themselves() -> None:
    key = ClaudeCodeApiKey(value="sk-ant-api03-key")

    assert await ClaudeCodeLogins().resolve(key, binary="/bin/claude") is key


@pytest.mark.asyncio
async def test_entering_the_scope_verifies_the_login_file(tmp_path: Path) -> None:
    binary = write_fake_claude(tmp_path)
    (tmp_path / "empty").mkdir()

    with pytest.raises(ClaudeCodeUnavailable, match="credentials file"):
        async with ctx.scope(
            "test",
            disposables=(
                ClaudeCode(
                    binary=binary,
                    credential=ClaudeCodeLocalLogin(config_directory=str(tmp_path / "empty")),
                ),
            ),
        ):
            pass

    directory = _login_directory(tmp_path)
    async with ctx.scope(
        "test",
        disposables=(
            ClaudeCode(
                binary=binary,
                credential=ClaudeCodeLocalLogin(config_directory=str(directory)),
            ),
        ),
    ):
        pass


@pytest.mark.asyncio
async def test_a_run_gets_the_access_token_and_its_scopes_only(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path)
    binary = write_fake_claude(tmp_path)
    record = tmp_path / "record.json"
    config = ClaudeCodeConfig(
        timeout=30,
        stop_grace=2,
        connectors=ClaudeCodeAccountConnectors(),
        env={"FAKE_CLAUDE_RECORD": str(record)},
    )

    async with ctx.scope(
        "test",
        config,
        disposables=(
            ClaudeCode(
                binary=binary,
                credential=ClaudeCodeLocalLogin(config_directory=str(directory)),
            ),
        ),
    ):
        _ = [chunk async for chunk in claude_code_agent("coder").call(input="hello")]

    recorded = json.loads(record.read_text())
    assert recorded["key"] == TOKEN
    assert recorded["variable"] == "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"
    assert recorded["environment"]["CLAUDE_CODE_OAUTH_SCOPES"] == "user:inference user:mcp_servers"
    assert recorded["environment"]["MCP_CONNECTION_NONBLOCKING"] == "0"
    assert recorded["environment"]["CLAUDE_CONFIG_DIR"] != str(directory)
    assert "--strict-mcp-config" not in recorded["arguments"]
    assert TOKEN not in json.dumps(recorded["arguments"])
    assert REFRESH_TOKEN not in json.dumps(recorded)


@pytest.mark.asyncio
async def test_connectors_without_the_scope_stop_the_run(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, scopes=("user:inference",))
    binary = write_fake_claude(tmp_path)

    async with ctx.scope(
        "test",
        ClaudeCodeConfig(connectors=ClaudeCodeAccountConnectors()),
        disposables=(
            ClaudeCode(
                binary=binary,
                credential=ClaudeCodeLocalLogin(config_directory=str(directory)),
            ),
        ),
    ):
        with pytest.raises(ClaudeCodeUnavailable, match="mcp_servers"):
            _ = [chunk async for chunk in claude_code_agent("coder").call(input="hello")]


@pytest.mark.asyncio
@pytest.mark.parametrize(("source", "accepted"), [("none", True), ("ANTHROPIC_API_KEY", False)])
async def test_access_tokens_are_verified_by_the_none_source(
    tmp_path: Path,
    source: str,
    accepted: bool,
) -> None:
    directory = _login_directory(tmp_path)
    binary = write_fake_claude(tmp_path)
    config = ClaudeCodeConfig(timeout=30, stop_grace=2, env={"FAKE_CLAUDE_SOURCE": source})

    async with ctx.scope(
        "test",
        config,
        disposables=(
            ClaudeCode(
                binary=binary,
                credential=ClaudeCodeLocalLogin(config_directory=str(directory)),
            ),
        ),
    ):
        if accepted:
            _ = [chunk async for chunk in claude_code_agent("coder").call(input="hello")]

        else:
            with pytest.raises(ClaudeCodeUnavailable, match="access token of a login"):
                _ = [chunk async for chunk in claude_code_agent("coder").call(input="hello")]
