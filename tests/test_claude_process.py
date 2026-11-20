import asyncio
import json
import os
import subprocess
import time
from contextlib import aclosing
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from draive.claude import (
    ClaudeCodeFailed,
    ClaudeCodeResult,
    ClaudeCodeStarted,
    ClaudeCodeText,
    ClaudeCodeToolResult,
    ClaudeCodeToolUse,
    ClaudeCodeUnavailable,
)
from draive.claude import process as process_module
from draive.claude.launch import ClaudeCodeSession
from draive.claude.process import marked_processes, run_process
from tests._claude_cli import (
    alive,
    environment_listing_available,
    launch_for,
    write_fake_claude,
)

listing_required = pytest.mark.skipif(
    not environment_listing_available(),
    reason="the platform does not list the environments of processes",
)

SECRET = "sk-ant-api03-secret-credential"


async def _events(launch: Any) -> list[Any]:
    return [event async for event in run_process(launch)]


def _recorded(directory: Path) -> Any:
    return json.loads((directory / "record.json").read_text())


async def _eventually(condition: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True

        await asyncio.sleep(0.05)

    return condition()


@pytest.mark.asyncio
async def test_run_streams_events_in_order(tmp_path: Path) -> None:
    events = await _events(launch_for(write_fake_claude(tmp_path), tmp_path, prompt="ping"))

    assert [type(event) for event in events] == [
        ClaudeCodeStarted,
        ClaudeCodeText,
        ClaudeCodeResult,
    ]
    started, text, result = events
    assert started.model == "fake-model"
    assert text.text == "echo: ping"
    assert result.failed is False
    assert result.cost_usd == 0.5


@pytest.mark.asyncio
async def test_tools_are_reported_between_texts(tmp_path: Path) -> None:
    events = await _events(launch_for(write_fake_claude(tmp_path), tmp_path, "tools"))

    assert [type(event) for event in events] == [
        ClaudeCodeStarted,
        ClaudeCodeText,
        ClaudeCodeToolUse,
        ClaudeCodeToolResult,
        ClaudeCodeText,
        ClaudeCodeResult,
    ]


@pytest.mark.asyncio
async def test_the_prompt_is_sent_through_the_input_and_not_the_arguments(tmp_path: Path) -> None:
    await _events(launch_for(write_fake_claude(tmp_path), tmp_path, prompt="PROMPT-MARKER"))

    recorded = _recorded(tmp_path)
    assert "PROMPT-MARKER" not in " ".join(recorded["arguments"])
    assert json.loads(recorded["line"])["message"]["content"] == [
        {"type": "text", "text": "PROMPT-MARKER"}
    ]
    assert recorded["directory"].endswith("work")


@pytest.mark.asyncio
async def test_the_credential_is_handed_over_through_a_descriptor_only(tmp_path: Path) -> None:
    await _events(launch_for(write_fake_claude(tmp_path), tmp_path))

    recorded = _recorded(tmp_path)
    assert recorded["key"] == SECRET
    assert recorded["variable"] == "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"
    assert recorded["environment"][recorded["variable"]].isdigit()
    assert SECRET not in json.dumps(recorded["arguments"])
    assert SECRET not in json.dumps(recorded["environment"])


@pytest.mark.asyncio
async def test_oauth_tokens_use_their_own_descriptor(tmp_path: Path) -> None:
    launch = launch_for(write_fake_claude(tmp_path), tmp_path, credential=None)
    launch = launch.updating(
        credential="sk-ant-oat-token",
        credential_variable="CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
    )

    await _events(launch)

    recorded = _recorded(tmp_path)
    assert recorded["key"] == "sk-ant-oat-token"
    assert recorded["variable"] == "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"


@pytest.mark.asyncio
async def test_runs_without_a_credential_have_no_descriptor(tmp_path: Path) -> None:
    await _events(launch_for(write_fake_claude(tmp_path), tmp_path, credential=None))

    recorded = _recorded(tmp_path)
    assert recorded["key"] is None
    assert recorded["variable"] is None


@pytest.mark.asyncio
async def test_the_parent_environment_does_not_reach_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the parent is typically Claude Code itself
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "parent-key")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://parent-collector")
    monkeypatch.setenv("SOME_PARENT_SECRET", "parent-secret")

    await _events(launch_for(write_fake_claude(tmp_path), tmp_path))

    environment = _recorded(tmp_path)["environment"]
    for name in (
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "ANTHROPIC_API_KEY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "SOME_PARENT_SECRET",
    ):
        assert name not in environment

    assert environment["HOME"] == str(tmp_path)
    assert environment["CLAUDE_CONFIG_DIR"] == str(tmp_path)
    assert environment["PATH"] == os.environ["PATH"]


@pytest.mark.asyncio
async def test_sessions_are_passed_through_the_arguments(tmp_path: Path) -> None:
    binary = write_fake_claude(tmp_path)

    new = await _events(
        launch_for(binary, tmp_path, session=ClaudeCodeSession(mode="new", identifier="fixed-id"))
    )
    assert new[0].session_id == "fixed-id"

    resumed = await _events(
        launch_for(binary, tmp_path, session=ClaudeCodeSession(mode="resume", identifier="old-id"))
    )
    assert resumed[0].session_id == "old-id"


@pytest.mark.asyncio
async def test_lines_larger_than_the_default_stream_limit_are_read(tmp_path: Path) -> None:
    events = await _events(launch_for(write_fake_claude(tmp_path), tmp_path, "bigline"))

    result = next(event for event in events if isinstance(event, ClaudeCodeToolResult))
    assert len(result.content) == 300_000
    assert isinstance(events[-1], ClaudeCodeResult)


@pytest.mark.asyncio
async def test_a_flood_of_error_output_does_not_block_the_process(tmp_path: Path) -> None:
    events = await asyncio.wait_for(
        _events(launch_for(write_fake_claude(tmp_path), tmp_path, "stderr_flood")),
        timeout=20,
    )

    assert isinstance(events[-1], ClaudeCodeResult)
    assert events[-1].text == "survived"


@pytest.mark.asyncio
async def test_lines_which_are_not_events_are_skipped(tmp_path: Path) -> None:
    events = await _events(launch_for(write_fake_claude(tmp_path), tmp_path, "garbage"))

    assert [type(event) for event in events] == [
        ClaudeCodeStarted,
        ClaudeCodeText,
        ClaudeCodeResult,
    ]


@pytest.mark.asyncio
async def test_a_failed_turn_is_reported_by_its_result(tmp_path: Path) -> None:
    events = await _events(launch_for(write_fake_claude(tmp_path), tmp_path, "error401"))

    # the synthetic error message is not text, the failure is the result
    assert [type(event) for event in events] == [ClaudeCodeStarted, ClaudeCodeResult]
    result = events[-1]
    assert (result.failed, result.api_error_status, result.terminal_reason) == (
        True,
        401,
        "api_error",
    )


@pytest.mark.asyncio
async def test_a_process_ending_without_a_result_fails_with_redacted_error_output(
    tmp_path: Path,
) -> None:
    launch = launch_for(write_fake_claude(tmp_path), tmp_path, "noresult")

    with pytest.raises(ClaudeCodeFailed, match="without a result") as raised:
        await _events(launch)

    assert raised.value.exit_code == 3
    assert "fatal: bad credentials" in raised.value.stderr
    assert SECRET not in raised.value.stderr
    assert "***" in raised.value.stderr
    assert SECRET not in str(raised.value)


@pytest.mark.asyncio
async def test_a_missing_executable_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ClaudeCodeUnavailable, match="failed to start"):
        await _events(launch_for(str(tmp_path / "missing"), tmp_path))


@pytest.mark.asyncio
async def test_a_missing_working_directory_is_unavailable(tmp_path: Path) -> None:
    launch = launch_for(write_fake_claude(tmp_path), tmp_path).updating(
        directory=str(tmp_path / "missing")
    )

    with pytest.raises(ClaudeCodeUnavailable):
        await _events(launch)


@pytest.mark.asyncio
async def test_the_deadline_stops_the_process_and_its_children(tmp_path: Path) -> None:
    launch = launch_for(write_fake_claude(tmp_path), tmp_path, "hang", timeout=1.0)
    started = time.monotonic()

    with pytest.raises(ClaudeCodeFailed, match="timed out after 1 seconds"):
        await _events(launch)

    assert time.monotonic() - started < 10
    child = int((tmp_path / "child.pid").read_text().split()[0])
    assert await _eventually(lambda: not alive(child))


@pytest.mark.asyncio
async def test_closing_the_stream_stops_the_process_and_its_children(tmp_path: Path) -> None:
    stream = run_process(launch_for(write_fake_claude(tmp_path), tmp_path, "hang"))
    try:
        async for event in stream:
            if isinstance(event, ClaudeCodeText):
                break  # the consumer is done before the end of the run

    finally:
        await stream.aclose()

    child = int((tmp_path / "child.pid").read_text().split()[0])
    assert await _eventually(lambda: not alive(child))


@pytest.mark.asyncio
async def test_cancellation_stops_the_process_and_its_children(tmp_path: Path) -> None:
    reading = asyncio.Event()

    async def consume() -> None:
        async for event in run_process(launch_for(write_fake_claude(tmp_path), tmp_path, "hang")):
            if isinstance(event, ClaudeCodeText):
                reading.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(reading.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    child = int((tmp_path / "child.pid").read_text().split()[0])
    assert await _eventually(lambda: not alive(child))


@pytest.mark.asyncio
async def test_a_process_ignoring_termination_is_killed_after_the_grace(tmp_path: Path) -> None:
    launch = launch_for(
        write_fake_claude(tmp_path),
        tmp_path,
        "hang",
        grace=0.5,
        env={"FAKE_CLAUDE_STUBBORN": "1"},
    )
    stream = run_process(launch)
    try:
        async for event in stream:
            if isinstance(event, ClaudeCodeText):
                break

    finally:
        started = time.monotonic()
        await stream.aclose()

    elapsed = time.monotonic() - started
    assert 0.4 <= elapsed < 10  # killed after the grace, not right away
    child = int((tmp_path / "child.pid").read_text().split()[0])
    assert await _eventually(lambda: not alive(child))


@pytest.mark.asyncio
async def test_a_consumer_failure_stops_the_process(tmp_path: Path) -> None:
    # a failure within the loop does not close the stream, its owner does
    with pytest.raises(RuntimeError, match="consumer"):
        async with aclosing(
            run_process(launch_for(write_fake_claude(tmp_path), tmp_path, "hang"))
        ) as stream:
            async for event in stream:
                if isinstance(event, ClaudeCodeText):
                    raise RuntimeError("consumer failed")

    (child,) = _children(tmp_path)
    assert await _eventually(lambda: not alive(child))


def _children(tmp_path: Path) -> list[int]:
    path = tmp_path / "child.pid"
    # not written when the fake failed before, which is what a test should report
    return [int(pid) for pid in path.read_text().split()] if path.exists() else []


@listing_required
@pytest.mark.asyncio
async def test_children_outliving_a_finished_run_are_stopped(tmp_path: Path) -> None:
    events = await _events(launch_for(write_fake_claude(tmp_path), tmp_path, "orphans"))

    assert isinstance(events[-1], ClaudeCodeResult)
    children = _children(tmp_path)
    assert len(children) == 2
    for child in children:  # one in the group of the process, one which left it
        assert await _eventually(lambda child=child: not alive(child))


@listing_required
@pytest.mark.asyncio
async def test_children_detached_from_the_process_group_stop_with_the_deadline(
    tmp_path: Path,
) -> None:
    launch = launch_for(
        write_fake_claude(tmp_path),
        tmp_path,
        "hang",
        timeout=1.0,
        env={"FAKE_CLAUDE_DETACH": "1"},
    )

    with pytest.raises(ClaudeCodeFailed, match="timed out"):
        await _events(launch)

    (child,) = _children(tmp_path)
    assert await _eventually(lambda: not alive(child))


@listing_required
@pytest.mark.asyncio
async def test_children_detached_from_the_process_group_stop_when_the_stream_is_closed(
    tmp_path: Path,
) -> None:
    stream = run_process(
        launch_for(write_fake_claude(tmp_path), tmp_path, "hang", env={"FAKE_CLAUDE_DETACH": "1"})
    )
    try:
        async for event in stream:
            if isinstance(event, ClaudeCodeText):
                break

    finally:
        await stream.aclose()

    (child,) = _children(tmp_path)
    assert await _eventually(lambda: not alive(child))


@pytest.mark.asyncio
async def test_unrelated_processes_are_left_alone(tmp_path: Path) -> None:
    bystander = await asyncio.create_subprocess_exec("sleep", "300")
    try:
        await _events(launch_for(write_fake_claude(tmp_path), tmp_path, "orphans"))
        assert await _eventually(lambda: not alive(_children(tmp_path)[0]))

        assert bystander.returncode is None
        assert alive(bystander.pid)

    finally:
        bystander.kill()
        await bystander.wait()
        for child in _children(tmp_path):  # whatever could not be tracked on this platform
            try:
                os.kill(child, 9)

            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_runs_are_marked_apart_from_each_other(tmp_path: Path) -> None:
    binary = write_fake_claude(tmp_path)
    first = launch_for(binary, tmp_path)
    second = launch_for(binary, tmp_path)

    await _events(first)

    assert first.marker and first.marker != second.marker
    assert _recorded(tmp_path)["environment"]["DRAIVE_CLAUDE_RUN"] == first.marker


@pytest.mark.asyncio
async def test_the_deadline_terminates_gracefully_before_it_kills(tmp_path: Path) -> None:
    flag = tmp_path / "terminated.flag"
    launch = launch_for(
        write_fake_claude(tmp_path),
        tmp_path,
        "hang",
        timeout=1.0,
        grace=5.0,
        env={"FAKE_CLAUDE_GRACEFUL": str(flag)},
    )
    started = time.monotonic()

    with pytest.raises(ClaudeCodeFailed, match="timed out"):
        await _events(launch)

    assert flag.read_text() == "terminated"  # it had the chance to stop its own children
    assert time.monotonic() - started < 4  # and did not need the grace


@pytest.mark.asyncio
async def test_the_deadline_kills_a_process_which_does_not_terminate(tmp_path: Path) -> None:
    launch = launch_for(
        write_fake_claude(tmp_path),
        tmp_path,
        "hang",
        timeout=1.0,
        grace=0.5,
        env={"FAKE_CLAUDE_STUBBORN": "1"},
    )
    started = time.monotonic()

    with pytest.raises(ClaudeCodeFailed, match="timed out"):
        await _events(launch)

    assert 1.4 <= time.monotonic() - started < 6  # the deadline and then the grace
    (child,) = _children(tmp_path)
    assert await _eventually(lambda: not alive(child))


@listing_required
def test_processes_are_found_by_the_marker_of_their_run() -> None:
    marker = uuid4().hex
    child = subprocess.Popen(
        ["sleep", "30"],
        env={**os.environ, "DRAIVE_CLAUDE_RUN": marker},
    )
    try:
        time.sleep(0.3)
        assert marked_processes(marker) == [child.pid]
        assert marked_processes(uuid4().hex) == []

    finally:
        child.kill()
        child.wait()


@pytest.mark.asyncio
async def test_a_consumer_holding_the_result_past_the_deadline_is_not_failed(
    tmp_path: Path,
) -> None:
    launch = launch_for(write_fake_claude(tmp_path), tmp_path, "echo", timeout=0.8)
    events = []

    async for event in run_process(launch):
        events.append(event)
        if isinstance(event, ClaudeCodeResult):
            await asyncio.sleep(1.5)  # the deadline passes while the consumer holds the result

    assert isinstance(events[-1], ClaudeCodeResult)


@pytest.mark.asyncio
async def test_the_kill_of_the_deadline_is_not_sent_after_the_process_is_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    original = process_module._signal_group

    def recording(process: Any, signum: int) -> None:
        signals.append(signum)
        original(process, signum)

    monkeypatch.setattr(process_module, "_signal_group", recording)
    launch = launch_for(
        write_fake_claude(tmp_path),
        tmp_path,
        "hang",
        timeout=0.5,
        grace=1.0,
        env={"FAKE_CLAUDE_GRACEFUL": str(tmp_path / "terminated.flag")},
    )

    with pytest.raises(ClaudeCodeFailed, match="timed out"):
        await _events(launch)

    sent = len(signals)
    await asyncio.sleep(1.5)  # the grace passes, a group identifier could be someone else's now
    assert len(signals) == sent


@pytest.mark.asyncio
async def test_the_session_reported_before_a_crash_is_kept_in_the_failure(tmp_path: Path) -> None:
    launch = launch_for(
        write_fake_claude(tmp_path),
        tmp_path,
        "init_then_exit",
        session=ClaudeCodeSession(mode="new", identifier="reported-session"),
    )

    with pytest.raises(ClaudeCodeFailed, match="without a result") as raised:
        await _events(launch)

    assert raised.value.session_id == "reported-session"
    assert raised.value.exit_code == 4


@pytest.mark.asyncio
async def test_credentials_are_redacted_across_the_boundaries_of_the_error_output(
    tmp_path: Path,
) -> None:
    launch = launch_for(write_fake_claude(tmp_path), tmp_path, "split_secret")

    with pytest.raises(ClaudeCodeFailed) as raised:
        await _events(launch)

    stderr = raised.value.stderr
    # written in two parts with the output cut to its tail between them
    assert SECRET[:10] not in stderr
    assert SECRET[10:] not in stderr
    assert "***" in stderr
    assert len(stderr) < 9000  # bounded


@pytest.mark.asyncio
async def test_waiting_for_the_error_output_does_not_swallow_cancellation() -> None:
    never = asyncio.create_task(asyncio.sleep(60))
    waiting = asyncio.create_task(process_module._settle(never))
    await asyncio.sleep(0.1)

    waiting.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiting

    never.cancel()


@pytest.mark.asyncio
async def test_output_written_after_the_run_reaches_nobody(tmp_path: Path) -> None:
    problems: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: problems.append(context))
    try:
        events = await _events(launch_for(write_fake_claude(tmp_path), tmp_path, "late_output"))
        assert isinstance(events[-1], ClaudeCodeResult)

        await asyncio.sleep(3.5)  # a child which held the output writes now, the run is long over

    finally:
        loop.set_exception_handler(None)
        for child in _children(tmp_path):
            try:
                os.kill(child, 9)

            except ProcessLookupError:
                pass

    assert problems == []


@pytest.mark.asyncio
async def test_oversized_credentials_are_refused_before_anything_starts(tmp_path: Path) -> None:
    launch = launch_for(write_fake_claude(tmp_path), tmp_path, credential="k" * 5000)

    with pytest.raises(ClaudeCodeUnavailable, match="larger than"):
        await _events(launch)

    assert not (tmp_path / "record.json").exists()
