import asyncio
import os
import signal
import sys
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Final

from haiway import Missing

from draive.claude.events import (
    ClaudeCodeEvent,
    ClaudeCodeResult,
    ClaudeCodeStarted,
    parse_events,
)
from draive.claude.launch import RUN_MARKER_VARIABLE, ClaudeCodeLaunch
from draive.claude.types import ClaudeCodeFailed, ClaudeCodeUnavailable

__all__ = (
    "MAX_CREDENTIAL_BYTES",
    "marked_processes",
    "run_process",
)

# tool results are reported within a single output line, `StreamReader` would refuse
# lines above its limit - 64 KiB by default
_LINE_LIMIT: Final[int] = 64 * 1024 * 1024
_STDERR_TAIL: Final[int] = 8 * 1024
# the output of a process which exited is complete after this time, whoever else holds its pipe
_OUTPUT_DRAIN: Final[float] = 1.0
# the credential is written to the pipe before its reader exists, it has to fit within the
# capacity of a pipe - which POSIX requires to be larger than that - or the write would block
MAX_CREDENTIAL_BYTES: Final[int] = 4096


def _credential_pipe(
    credential: str,
    /,
) -> int:
    data: bytes = credential.encode()
    if len(data) > MAX_CREDENTIAL_BYTES:
        raise ClaudeCodeUnavailable(f"the credential is larger than {MAX_CREDENTIAL_BYTES} bytes")

    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, data)

    except BaseException:
        os.close(read_fd)  # the descriptor would not be closed by dropping its number
        raise

    finally:
        os.close(write_fd)

    return read_fd


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
        credential_fd = _credential_pipe(launch.credential)
        pass_fds = (credential_fd,)
        environment[launch.credential_variable] = str(credential_fd)

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

    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    stderr_tail: bytearray = bytearray()
    stderr_task: asyncio.Task[None] = asyncio.create_task(
        _drain(process, stderr_tail, launch.secrets)
    )
    exit_task: asyncio.Task[None] = asyncio.create_task(_end_output_after_exit(process))
    timed_out: bool = False
    escalation: asyncio.TimerHandle | None = None

    def deadline() -> None:
        nonlocal timed_out, escalation
        timed_out = True
        # terminated first, Claude Code stops the commands of its tools then - which a kill leaves
        # running - and killed only when it does not exit in time
        _signal_group(process, signal.SIGTERM)
        escalation = loop.call_later(launch.grace, _signal_group, process, signal.SIGKILL)

    deadline_handle: asyncio.TimerHandle = loop.call_later(launch.timeout, deadline)
    session_id: str | None = None
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
                    session_id=session_id,
                ) from exc

            if not line:
                break

            for event in parse_events(line):
                if isinstance(event, ClaudeCodeStarted):
                    session_id = event.session_id

                elif isinstance(event, ClaudeCodeResult):
                    result = event
                    # the work is done, a consumer taking its time must not be failed by the
                    # deadline which passes while it holds the result
                    deadline_handle.cancel()

                yield event

            if result is not None:
                break  # nothing follows the result, children keeping the output open do not matter

        try:
            exit_code: int | None = await asyncio.wait_for(process.wait(), launch.grace or None)

        except TimeoutError:
            exit_code = None  # the cleanup kills it

        if result is None and timed_out:
            raise ClaudeCodeFailed(
                f"timed out after {launch.timeout:g} seconds",
                session_id=session_id,
                exit_code=exit_code,
                stderr=_redacted(stderr_tail, launch.secrets),
            )

        if result is None:
            await _settle(stderr_task)
            raise ClaudeCodeFailed(
                "the process ended without a result",
                session_id=session_id,
                exit_code=exit_code,
                stderr=_redacted(stderr_tail, launch.secrets),
            )

    finally:
        deadline_handle.cancel()
        if escalation is not None:
            escalation.cancel()  # the cleanup takes over, a late kill could hit a reused group

        await _finish(process, stderr_task, exit_task, launch.grace, launch.marker)


async def _drain(
    process: asyncio.subprocess.Process,
    tail: bytearray,
    secrets: Sequence[str],
    /,
) -> None:
    # an unread error output fills its pipe and blocks the process, only the tail is kept
    assert process.stderr is not None  # nosec: B101
    hidden: Sequence[bytes] = tuple(secret.encode() for secret in secrets if secret)
    while chunk := await process.stderr.read(4096):
        tail.extend(chunk)
        # redacted before the tail is cut, which could leave a part of a secret behind
        for secret in hidden:
            if secret in tail:
                tail[:] = bytes(tail).replace(secret, b"***")

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
    _close_output(process)


def _close_output(
    process: asyncio.subprocess.Process,
    /,
) -> None:
    # closing the pipes ends the readers, whatever is written to them later reaches nobody
    transport: asyncio.SubprocessTransport | None = getattr(process, "_transport", None)
    if transport is None:
        return

    for fd in (1, 2):
        pipe = transport.get_pipe_transport(fd)
        if pipe is not None:
            pipe.close()


async def _settle(
    stderr_task: asyncio.Task[None],
    /,
) -> None:
    # the error output has ended with the process, let the tail be complete
    try:
        await asyncio.wait_for(asyncio.shield(stderr_task), 1.0)

    except TimeoutError:
        pass  # held open by a child, what was read is what there is


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

    Reads ``/proc``, where the environment of a process is exact. Platforms without it have no
    reliable source - a match within the command line of an unrelated process is not a reason to
    stop it - and nothing is found there.

    Parameters
    ----------
    marker
        Value of the run marker variable.

    Returns
    -------
    list[int]
        Identifiers of the processes, this one excluded.
    """
    proc: Path = Path("/proc")
    if not (proc / "self" / "environ").exists():
        return []

    needle: bytes = f"{RUN_MARKER_VARIABLE}={marker}".encode()
    found: list[int] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue

        try:
            if needle in (entry / "environ").read_bytes().split(b"\0"):
                found.append(int(entry.name))

        except OSError:
            continue  # gone or not ours

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

    _close_output(process)
    stderr_task.cancel()
    exit_task.cancel()
    await asyncio.gather(stderr_task, exit_task, return_exceptions=True)
