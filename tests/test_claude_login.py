import asyncio
import json
import os
import stat
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from haiway import ValidationError, ctx

from draive.claude import (
    ClaudeCode,
    ClaudeCodeAccessToken,
    ClaudeCodeAccountConnectors,
    ClaudeCodeApiKey,
    ClaudeCodeConfig,
    ClaudeCodeEvent,
    ClaudeCodeLaunch,
    ClaudeCodeLocalLogin,
    ClaudeCodeResolving,
    ClaudeCodeRunner,
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
time.sleep(1.0 if mode == "slow" else 0.3)
if mode in ("refresh", "slow"):
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


async def _never_running(launch: ClaudeCodeLaunch, /) -> AsyncGenerator[ClaudeCodeEvent]:
    raise AssertionError("not expected to run")
    yield  # pragma: no cover


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
async def test_a_refresh_which_hangs_is_not_waited_for_and_killed_at_its_limit(
    tmp_path: Path,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)
    (directory / "mode").write_text("hang")
    logins = ClaudeCodeLogins(refresh_timeout=0.2, refresh_limit=1.5)

    async with ctx.scope("test"):
        token = await logins.resolve(
            ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200),
            binary=_fake_binary(tmp_path),
        )
        await _until((directory / "pid").exists)
        pid = int((directory / "pid").read_text())
        assert alive(pid)  # the run went on, the refresh was left running

        await logins.drain()

    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == TOKEN  # still valid
    assert not alive(pid)
    assert _lock_is_free(directory)


@pytest.mark.asyncio
async def test_a_refresh_not_waited_for_still_completes_before_the_drain_returns(
    tmp_path: Path,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)
    logins = ClaudeCodeLogins(refresh_timeout=0.01)
    login = ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200)

    async with ctx.scope("test"):
        token = await logins.resolve(login, binary=_fake_binary(tmp_path))
        await logins.drain()

    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == TOKEN  # did not wait for the refresh
    assert REFRESHED in (directory / ".credentials.json").read_text()
    assert len(_invocations(directory)) == 1


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


def test_concurrent_resolutions_within_different_event_loops_refresh_once(
    tmp_path: Path,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=60_000)
    binary = _fake_binary(tmp_path)
    login = ClaudeCodeLocalLogin(config_directory=str(directory))
    start = threading.Barrier(3)

    async def resolving() -> str:
        async with ctx.scope("test"):
            token = await ClaudeCodeLogins().resolve(login, binary=binary)

        assert isinstance(token, ClaudeCodeAccessToken)
        return token.value

    def in_own_loop() -> str:
        start.wait()  # both loops see the token about to expire, only the file lock serializes
        return asyncio.run(resolving())

    with ThreadPoolExecutor(max_workers=3) as executor:
        values = [future.result() for future in [executor.submit(in_own_loop) for _ in range(3)]]

    assert values == [REFRESHED] * 3
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
@pytest.mark.parametrize(("expires_in_ms", "valid"), [(60_000, True), (-60_000, False)])
async def test_a_lock_held_for_too_long_fails_only_an_expired_token(
    tmp_path: Path,
    expires_in_ms: int,
    valid: bool,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=expires_in_ms)
    holder = _hold_lock(directory, 3)
    logins = ClaudeCodeLogins(lock_timeout=0.3)
    login = ClaudeCodeLocalLogin(config_directory=str(directory))

    try:
        async with ctx.scope("test"):
            if valid:
                token = await logins.resolve(login, binary=_fake_binary(tmp_path))
                assert isinstance(token, ClaudeCodeAccessToken)
                assert token.value == TOKEN

            else:
                with pytest.raises(ClaudeCodeUnavailable, match="for too long"):
                    await logins.resolve(login, binary=_fake_binary(tmp_path))

            await logins.drain()

    finally:
        holder.kill()
        holder.wait()

    assert _invocations(directory) == []


@pytest.mark.asyncio
@pytest.mark.skipif(os.geteuid() == 0, reason="root writes to read-only directories")
async def test_an_unusable_lock_fails_only_an_expired_token(tmp_path: Path) -> None:
    valid = _login_directory(tmp_path, name="valid", expires_in_ms=60_000)
    expired = _login_directory(tmp_path, name="expired", expires_in_ms=-60_000)
    logins = ClaudeCodeLogins()
    for directory in (valid, expired):
        directory.chmod(0o500)  # the lock file can not be created

    try:
        async with ctx.scope("test"):
            token = await logins.resolve(
                ClaudeCodeLocalLogin(config_directory=str(valid)),
                binary=_fake_binary(tmp_path),
            )
            assert isinstance(token, ClaudeCodeAccessToken)
            assert token.value == TOKEN

            with pytest.raises(ClaudeCodeUnavailable, match="lock of the login refresh"):
                await logins.resolve(
                    ClaudeCodeLocalLogin(config_directory=str(expired)),
                    binary=_fake_binary(tmp_path),
                )

    finally:
        for directory in (valid, expired):
            directory.chmod(0o700)


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
        await logins.drain()  # the refresh of the cancelled resolution, it may still run

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
    "too large": json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": TOKEN,
                "refreshToken": REFRESH_TOKEN,
                "expiresAt": 1,
                "scopes": [],
                "padding": "x" * 70_000,
            }
        }
    ).encode(),
    "nested deep": b'{"claudeAiOauth": {"refreshToken": "'
    + REFRESH_TOKEN.encode()
    + b'", "deep": '
    + b"[" * 30_000
    + b"]" * 30_000
    + b"}}",
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


def test_an_access_token_needs_a_timezone() -> None:
    with pytest.raises(ValidationError):
        ClaudeCodeAccessToken(value=TOKEN, scopes=(), expires_at=datetime.now())

    assert ClaudeCodeAccessToken(value=TOKEN, scopes=(), expires_at=datetime.now(UTC))


def test_a_runner_made_without_resolving_gets_its_own() -> None:
    runner = ClaudeCodeRunner(binary="/bin/claude", running=_never_running)

    assert isinstance(runner.resolving, ClaudeCodeResolving)
    assert runner.updating(binary="/bin/other").resolving is runner.resolving
    assert runner.to_mapping()["resolving"] == runner.resolving


def _combined_binary(root: Path) -> str:
    # `claude auth status` refreshes, anything else is a run
    refresh = _fake_binary(root)
    run = write_fake_claude(root)
    path = root / "claude-combined"
    path.write_text(
        f'#!/bin/sh\nif [ "$1" = auth ]; then exec {refresh} "$@"; fi\nexec {run} "$@"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.mark.asyncio
async def test_the_scope_waits_for_a_refresh_its_cancelled_run_left(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=60_000)
    binary = _combined_binary(tmp_path)

    async with ctx.scope(
        "test",
        ClaudeCodeConfig(timeout=30, stop_grace=2),
        disposables=(
            ClaudeCode(
                binary=binary,
                credential=ClaudeCodeLocalLogin(config_directory=str(directory)),
            ),
        ),
    ):

        async def running() -> None:
            _ = [chunk async for chunk in claude_code_agent("coder").call(input="hello")]

        run = asyncio.ensure_future(running())
        await _until(lambda: bool(_invocations(directory)))  # the refresh is running
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        assert REFRESHED not in (directory / ".credentials.json").read_text()

    # the refresh takes a while, it is written only because the exit waited for it
    assert REFRESHED in (directory / ".credentials.json").read_text()
    assert _lock_is_free(directory)


@pytest.mark.asyncio
async def test_every_entry_of_the_scope_drains_the_shared_logins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _login_directory(tmp_path)
    drained: list[ClaudeCodeLogins] = []
    original = ClaudeCodeLogins.drain

    async def recording(self: ClaudeCodeLogins) -> None:
        drained.append(self)
        await original(self)

    monkeypatch.setattr(ClaudeCodeLogins, "drain", recording)
    client = ClaudeCode(
        binary=write_fake_claude(tmp_path),
        credential=ClaudeCodeLocalLogin(config_directory=str(directory)),
    )

    async with ctx.scope("outer", disposables=(client,)):
        outer = ctx.state(ClaudeCodeRunner).resolving
        async with ctx.scope("inner", disposables=(client,)):
            assert ctx.state(ClaudeCodeRunner).resolving == outer

        assert [logins.resolve for logins in drained] == [outer]

    assert [logins.resolve for logins in drained] == [outer, outer]
    async with ctx.scope("again", disposables=(client,)):
        assert ctx.state(ClaudeCodeRunner).resolving != outer  # a fresh one after the last exit


@pytest.mark.asyncio
async def test_a_token_expiring_while_waiting_gets_the_refresh(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=500)
    (directory / "mode").write_text("slow")  # longer than the token lives

    async with ctx.scope("test"):
        token = await ClaudeCodeLogins(refresh_timeout=0.7).resolve(
            ClaudeCodeLocalLogin(config_directory=str(directory)),
            binary=_fake_binary(tmp_path),
        )

    assert isinstance(token, ClaudeCodeAccessToken)
    assert token.value == REFRESHED


@pytest.mark.asyncio
async def test_concurrent_runs_share_one_refresh_attempt(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)
    login = ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200)
    logins = ClaudeCodeLogins(lock_timeout=0.2, refresh_timeout=1, cooldown=0)
    holder = _hold_lock(directory, 2)

    try:
        async with ctx.scope("test"):
            tokens = await asyncio.gather(
                *(logins.resolve(login, binary=_fake_binary(tmp_path)) for _ in range(5))
            )
            await logins.drain()

    finally:
        holder.kill()
        holder.wait()

    assert {token.value for token in tokens if isinstance(token, ClaudeCodeAccessToken)} == {TOKEN}
    # one attempt waited for the lock and failed, every run shared it
    assert caplog.text.count("Refreshing the Claude Code login failed: ") == 1
    assert caplog.text.count("its current token is used") == 5


@pytest.mark.asyncio
async def test_a_failed_refresh_is_not_attempted_again_during_the_cooldown(
    tmp_path: Path,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)
    login = ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200)
    logins = ClaudeCodeLogins(lock_timeout=0.2)
    binary = _fake_binary(tmp_path)
    holder = _hold_lock(directory, 3)

    try:
        async with ctx.scope("test"):
            first = await logins.resolve(login, binary=binary)
            started = time.monotonic()
            second = await logins.resolve(login, binary=binary)
            elapsed = time.monotonic() - started

    finally:
        holder.kill()
        holder.wait()

    assert isinstance(first, ClaudeCodeAccessToken)
    assert isinstance(second, ClaudeCodeAccessToken)
    assert second.value == TOKEN
    assert elapsed < 0.15  # no second wait for the lock
    assert _invocations(directory) == []


@pytest.mark.asyncio
async def test_a_refresh_reports_to_its_scope_after_the_run_ended(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)
    (directory / "mode").write_text("nothing")
    login = ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200)

    async with ctx.scope("owner"):
        logins = ClaudeCodeLogins(refresh_timeout=0.05)
        async with ctx.scope("run"):
            await logins.resolve(login, binary=_fake_binary(tmp_path))

        caplog.clear()
        await logins.drain()  # the refresh ends after the run

    assert "did not extend its token" in caplog.text


@pytest.mark.asyncio
async def test_a_parser_running_out_of_stack_does_not_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = _login_directory(tmp_path)

    def exhausted(content: str) -> Any:
        held = content  # like the frames of the parser, holding the whole file
        raise RecursionError(f"maximum recursion depth exceeded {len(held)}")

    monkeypatch.setattr("draive.claude.login.json.loads", exhausted)

    with pytest.raises(ClaudeCodeUnavailable, match="malformed") as exc_info:
        await ClaudeCodeLogins().resolve(
            ClaudeCodeLocalLogin(config_directory=str(directory)), binary="/bin/claude"
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    for codeword in (REFRESH_TOKEN, TOKEN):
        assert not _holds(exc_info.value, codeword)


@pytest.mark.asyncio
async def test_a_cancelled_drain_does_not_cancel_the_refresh(tmp_path: Path) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=HOUR_MS)
    logins = ClaudeCodeLogins(refresh_timeout=0.01)
    login = ClaudeCodeLocalLogin(config_directory=str(directory), refresh_margin=7200)

    async with ctx.scope("test"):
        await logins.resolve(login, binary=_fake_binary(tmp_path))
        drain = asyncio.ensure_future(logins.drain())
        await asyncio.sleep(0.05)
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain

        await logins.drain()

    assert REFRESHED in (directory / ".credentials.json").read_text()


@pytest.mark.asyncio
async def test_static_credentials_are_resolved_to_themselves() -> None:
    key = ClaudeCodeApiKey(value="sk-ant-api03-key")

    assert await ClaudeCodeLogins().resolve(key, binary="/bin/claude") is key


@pytest.mark.asyncio
@pytest.mark.parametrize("keychain", [True, False])
async def test_an_expired_file_is_refused_at_startup_only_where_logins_are_in_the_keychain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keychain: bool,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=-HOUR_MS)
    monkeypatch.setattr("draive.claude.login._keychain_platform", lambda: keychain)
    login = ClaudeCodeLocalLogin(config_directory=str(directory))

    if keychain:
        with pytest.raises(ClaudeCodeUnavailable, match="keychain") as exc_info:
            await ClaudeCodeLogins.verify(login)

        assert "has expired" in str(exc_info.value)
        assert not _holds(exc_info.value, TOKEN)
        assert not _holds(exc_info.value, REFRESH_TOKEN)

    else:
        await ClaudeCodeLogins.verify(login)  # refreshed when a run needs it


@pytest.mark.asyncio
@pytest.mark.parametrize("keychain", [True, False])
async def test_an_expired_login_mentions_the_keychain_only_where_it_is_used(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keychain: bool,
) -> None:
    directory = _login_directory(tmp_path, expires_in_ms=-60_000)
    (directory / "mode").write_text("nothing")
    monkeypatch.setattr("draive.claude.login._keychain_platform", lambda: keychain)

    async with ctx.scope("test"):
        with pytest.raises(ClaudeCodeUnavailable, match="expired") as exc_info:
            await ClaudeCodeLogins().resolve(
                ClaudeCodeLocalLogin(config_directory=str(directory)),
                binary=_fake_binary(tmp_path),
            )

    assert ("keychain" in str(exc_info.value)) == keychain


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
