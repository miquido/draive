import asyncio
import os
import signal
import subprocess  # nosec: B404
import sys
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Final

from haiway import Missing

from draive.claude.events import ClaudeCodeEvent, ClaudeCodeResult, parse_events
from draive.claude.launch import RUN_MARKER_VARIABLE, ClaudeCodeLaunch
from draive.claude.types import ClaudeCodeFailed, ClaudeCodeUnavailable

__all__ = ("run_process",)

# tool results are reported within a single output line, `StreamReader` would refuse
# lines above its limit - 64 KiB by default
_LINE_LIMIT: Final[int] = 64 * 1024 * 1024
_STDERR_TAIL: Final[int] = 8 * 1024
# the output of a process which exited is complete after this time, whoever else holds its pipe
_OUTPUT_DRAIN: Final[float] = 1.0


async def run_process(  # noqa: C901, PLR0912, PLR0915
    launch: ClaudeCodeLaunch,
    /,
) -> AsyncGenerator[ClaudeCodeEvent]:
    """Run the Claude Code process and stream the events of its output.

    The process runs in its own session with the credential handed over through a file
    descriptor only, its output is read line by line and its error output kept as a bounded
    tail. Stopping the stream, i.e. closing it or failing, terminates the whole process group,
    and the run is limited by the deadline of the launch.

    Parameters
    ----------
    launch
        Prepared launch of the process.

    Yields
    ------
    ClaudeCodeEvent
        Events in the order of the output, the result is the last one.

    Raises
    ------
    ClaudeCodeUnavailable
        When the process can not be started.
    ClaudeCodeFailed
        When the process timed out or ended without reporting a result.
    """
    if sys.platform == "win32":
        raise ClaudeCodeUnavailable("Claude Code runs require a POSIX platform")

    environment: dict[str, str] = dict(launch.environment)
    pass_fds: tuple[int, ...] = ()
    credential_fd: int | None = None
    if launch.credential is not Missing:
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, launch.credential.encode())

        finally:
            os.close(write_fd)

        credential_fd = read_fd
        pass_fds = (read_fd,)
        environment[launch.credential_variable] = str(read_fd)

    try:
        process: asyncio.subprocess.Process = await asyncio.create_subprocess_exec(
            launch.binary,
            *launch.arguments,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=launch.directory,
            env=environment,
            start_new_session=True,  # own process group, stopped together with its children
            limit=_LINE_LIMIT,
            pass_fds=pass_fds,
        )

    except OSError as exc:
        raise ClaudeCodeUnavailable(
            f"failed to start '{launch.binary}' - {type(exc).__name__}: {exc.strerror or exc}"
        ) from exc

    finally:
        if credential_fd is not None:
            os.close(credential_fd)

    stderr_tail: bytearray = bytearray()
    stderr_task: asyncio.Task[None] = asyncio.create_task(_drain(process, stderr_tail))
    exit_task: asyncio.Task[None] = asyncio.create_task(_end_output_after_exit(process))
    timed_out: bool = False

    def deadline() -> None:
        nonlocal timed_out
        timed_out = True
        # terminated first, Claude Code stops the commands of its tools then - which a kill leaves
        # running - and killed only when it does not exit in time
        _signal_group(process, signal.SIGTERM)
        asyncio.get_running_loop().call_later(launch.grace, _signal_group, process, signal.SIGKILL)

    deadline_handle: asyncio.TimerHandle = asyncio.get_running_loop().call_later(
        launch.timeout, deadline
    )
    result: ClaudeCodeResult | None = None
    try:
        assert process.stdin is not None  # nosec: B101
        assert process.stdout is not None  # nosec: B101
        try:
            # the only message, the process ends its work with the turn
            process.stdin.write(launch.prompt.encode() + b"\n")
            await process.stdin.drain()
            process.stdin.close()

        except BrokenPipeError, ConnectionResetError:
            pass  # the process ended already, its output tells why

        while True:
            try:
                line: bytes = await process.stdout.readline()

            except ValueError as exc:  # a line above the limit
                raise ClaudeCodeFailed(
                    "the output line is too large",
                    session_id=result.session_id if result else None,
                ) from exc

            if not line:
                break

            for event in parse_events(line):
                if isinstance(event, ClaudeCodeResult):
                    result = event

                yield event

            if result is not None:
                break  # nothing follows the result, children keeping the output open do not matter

        try:
            exit_code: int | None = await asyncio.wait_for(process.wait(), launch.grace or None)

        except TimeoutError:
            exit_code = None  # the cleanup kills it

        if timed_out:
            raise ClaudeCodeFailed(
                f"timed out after {launch.timeout:g} seconds",
                session_id=result.session_id if result else None,
                exit_code=exit_code,
                stderr=_redacted(stderr_tail, launch.secrets),
            )

        if result is None:
            await _settle(stderr_task)
            raise ClaudeCodeFailed(
                "the process ended without a result",
                exit_code=exit_code,
                stderr=_redacted(stderr_tail, launch.secrets),
            )

    finally:
        deadline_handle.cancel()
        await _finish(process, stderr_task, exit_task, launch.grace, launch.marker)


async def _drain(
    process: asyncio.subprocess.Process,
    tail: bytearray,
    /,
) -> None:
    # an unread error output fills its pipe and blocks the process, only the tail is kept
    assert process.stderr is not None  # nosec: B101
    while chunk := await process.stderr.read(4096):
        tail.extend(chunk)
        if len(tail) > _STDERR_TAIL:
            del tail[: len(tail) - _STDERR_TAIL]


async def _end_output_after_exit(
    process: asyncio.subprocess.Process,
    /,
) -> None:
    # a child of the process which inherited its output keeps the pipe open after the exit,
    # the reader would wait for it - and for the deadline of the run - without an end
    await process.wait()
    await asyncio.sleep(_OUTPUT_DRAIN)
    assert process.stdout is not None  # nosec: B101
    process.stdout.feed_eof()


async def _settle(
    stderr_task: asyncio.Task[None],
    /,
) -> None:
    # the error output has ended with the process, let the tail be complete
    try:
        await asyncio.wait_for(asyncio.shield(stderr_task), 1.0)

    except TimeoutError, asyncio.CancelledError:
        pass


def _redacted(
    tail: bytearray,
    secrets: Sequence[str],
    /,
) -> str:
    text: str = tail.decode(errors="replace").strip()
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")

    return text


def marked_processes(
    marker: str,
    /,
) -> list[int]:
    """Find processes carrying the marker of a run within their environment.

    Reads ``/proc`` where there is one and falls back to the environments listed by ``ps``,
    which some platforms do not show - there is nothing to find on them.

    Parameters
    ----------
    marker
        Value of the run marker variable.

    Returns
    -------
    list[int]
        Identifiers of the processes, this one excluded.
    """
    needle: bytes = f"{RUN_MARKER_VARIABLE}={marker}".encode()
    if (Path("/proc") / "self" / "environ").exists():
        return _marked_in_proc(needle)

    return _marked_in_ps(needle)


def _marked_in_proc(
    needle: bytes,
    /,
) -> list[int]:
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue

        try:
            if needle in (entry / "environ").read_bytes().split(b"\0"):
                found.append(int(entry.name))

        except OSError:
            continue  # gone or not ours

    return found


def _marked_in_ps(
    needle: bytes,
    /,
) -> list[int]:
    try:
        output: bytes = subprocess.run(  # nosec: B603, B607
            ["ps", "-axeww", "-o", "pid=,command="],
            capture_output=True,
            timeout=10,
            check=False,
        ).stdout

    except OSError, subprocess.SubprocessError:
        return []

    found: list[int] = []
    for line in output.splitlines():
        if needle not in line:
            continue

        try:
            pid: int = int(line.split(None, 1)[0])

        except ValueError, IndexError:
            continue

        if pid != os.getpid():
            found.append(pid)

    return found


async def _reap(
    marker: str,
    /,
) -> None:
    # commands of the tools run in their own sessions, signals of the group do not reach them,
    # what they start inherits the marker of the run though, also when it is orphaned
    for pid in await asyncio.to_thread(marked_processes, marker):
        try:
            os.kill(pid, signal.SIGKILL)

        except ProcessLookupError, PermissionError:
            continue


def _signal_group(
    process: asyncio.subprocess.Process,
    signum: int,
    /,
) -> None:
    try:
        os.killpg(process.pid, signum)

    except ProcessLookupError, PermissionError:
        pass  # the group is gone already


async def _finish(
    process: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[None],
    exit_task: asyncio.Task[None],
    grace: float,
    marker: str,
    /,
) -> None:
    # cleanup has to complete however the run ended, cancellation included
    task: asyncio.Task[None] = asyncio.create_task(
        _dispose(process, stderr_task, exit_task, grace, marker)
    )
    cancelled: bool = False
    while not task.done():
        try:
            await asyncio.shield(task)

        except asyncio.CancelledError:
            cancelled = True

    if cancelled:
        raise asyncio.CancelledError

    task.result()


async def _dispose(
    process: asyncio.subprocess.Process,
    stderr_task: asyncio.Task[None],
    exit_task: asyncio.Task[None],
    grace: float,
    marker: str,
    /,
) -> None:
    if process.returncode is None:
        _signal_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), grace)

        except TimeoutError:
            _signal_group(process, signal.SIGKILL)
            await process.wait()

    # whatever the process left running, i.e. children of its tools
    _signal_group(process, signal.SIGKILL)
    if marker:
        await _reap(marker)

    if process.stdin is not None and not process.stdin.is_closing():
        process.stdin.close()

    stderr_task.cancel()
    exit_task.cancel()
    await asyncio.gather(stderr_task, exit_task, return_exceptions=True)
