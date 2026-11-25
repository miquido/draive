import asyncio
import os
import signal
import sys
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Final, final

from haiway import Missing, State

from draive.claude.config import RUN_MARKER_VARIABLE
from draive.claude.events import (
    ClaudeCodeEvent,
    ClaudeCodeResult,
    ClaudeCodeStarted,
    parse_events,
)
from draive.claude.launch import ClaudeCodeLaunch
from draive.claude.types import (
    ClaudeCodeEnded,
    ClaudeCodeFailed,
    ClaudeCodeOutputTooLarge,
    ClaudeCodeTimedOut,
    ClaudeCodeUnavailable,
    ClaudeCodeUnfinished,
)

__all__ = (
    "marked_processes",
    "run_process",
)

# tool results are reported within a single output line, `StreamReader` would refuse
# lines above its limit - 64 KiB by default
_LINE_LIMIT: Final[int] = 64 * 1024 * 1024
_STDERR_TAIL: Final[int] = 8 * 1024
# the output of a process which exited is complete after this time, whoever else holds its pipe
_OUTPUT_DRAIN: Final[float] = 1.0


@final
class _Observed(State):
    """What was seen of a run so far, the only source of its session and its result."""

    started: str | None = None
    result: ClaudeCodeResult | None = None

    @property
    def session_id(self) -> str | None:
        return (self.result.session_id if self.result is not None else None) or self.started

    def after(
        self,
        event: ClaudeCodeEvent,
        /,
    ) -> _Observed:
        match event:
            case ClaudeCodeStarted():
                return _Observed(started=event.session_id, result=self.result)

            case ClaudeCodeResult():
                return _Observed(started=self.started, result=event)

            case _:
                return self


@final
class _Deadline:
    """Limit of the duration of a run, terminating the process and killing it after the grace."""

    __slots__ = ("_escalation", "_expired", "_grace", "_process", "_timer")

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        timeout: float,
        grace: float,
    ) -> None:
        self._process: asyncio.subprocess.Process = process
        self._grace: float = grace
        self._expired: bool = False
        self._escalation: asyncio.TimerHandle | None = None
        self._timer: asyncio.TimerHandle = asyncio.get_running_loop().call_later(
            timeout, self._expire
        )

    @property
    def expired(self) -> bool:
        return self._expired

    def _expire(self) -> None:
        self._expired = True
        # terminated first, Claude Code stops the commands of its tools then - which a kill leaves
        # running - and killed only when it does not exit in time
        _signal_group(self._process, signal.SIGTERM)
        self._escalation = asyncio.get_running_loop().call_later(
            self._grace, _signal_group, self._process, signal.SIGKILL
        )

    def disarm(self) -> None:
        """The work is done, a consumer taking its time must not be failed by the limit."""
        self._timer.cancel()

    def release(self) -> None:
        """The cleanup takes over, a late kill could hit a group which was reused."""
        self._timer.cancel()
        if self._escalation is not None:
            self._escalation.cancel()


def _credential_pipe(
    credential: str,
    /,
) -> int:
    # the size of a credential is bounded by its type to fit the capacity of a pipe
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, credential.encode())

    except BaseException:
        os.close(read_fd)  # the descriptor would not be closed by dropping its number
        raise

    finally:
        os.close(write_fd)

    return read_fd


async def _spawn(
    launch: ClaudeCodeLaunch,
    /,
) -> asyncio.subprocess.Process:
    environment: dict[str, str] = {**launch.environment, RUN_MARKER_VARIABLE: launch.marker}
    descriptors: tuple[int, ...] = ()
    if launch.credential is not Missing:
        descriptor: int = _credential_pipe(launch.credential.value)
        environment[launch.credential.descriptor] = str(descriptor)
        descriptors = (descriptor,)

    try:
        return await asyncio.create_subprocess_exec(
            launch.binary,
            *launch.arguments,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=launch.directory,
            env=environment,
            start_new_session=True,  # own process group, stopped together with its children
            limit=_LINE_LIMIT,
            pass_fds=descriptors,
        )

    except OSError as exc:
        raise ClaudeCodeUnavailable(
            f"failed to start '{launch.binary}' - {type(exc).__name__}: {exc.strerror or exc}"
        ) from exc

    finally:
        for descriptor in descriptors:
            os.close(descriptor)


async def _send(
    process: asyncio.subprocess.Process,
    prompt: str,
    /,
) -> None:
    assert process.stdin is not None  # nosec: B101
    try:
        # the only message, the process ends its work with the turn
        process.stdin.write(prompt.encode() + b"\n")
        await process.stdin.drain()
        process.stdin.close()

    except BrokenPipeError, ConnectionResetError:
        pass  # the process ended already, its output tells why


async def _read_line(
    process: asyncio.subprocess.Process,
    observed: _Observed,
    /,
) -> bytes:
    assert process.stdout is not None  # nosec: B101
    try:
        return await process.stdout.readline()

    except ValueError as exc:  # a line above the limit
        raise ClaudeCodeOutputTooLarge(session_id=observed.session_id) from exc


async def _exit_code(
    process: asyncio.subprocess.Process,
    grace: float,
    /,
) -> int | None:
    # a process which closed its output exits right after, unless it is stuck - which is not to
    # hold the run back until its deadline, also when there is no grace to stop it
    try:
        return await asyncio.wait_for(process.wait(), max(grace, _OUTPUT_DRAIN))

    except TimeoutError:
        return None  # the cleanup kills it


async def _ended(
    launch: ClaudeCodeLaunch,
    deadline: _Deadline,
    observed: _Observed,
    exit_code: int | None,
    stderr_task: asyncio.Task[None],
    stderr_tail: bytearray,
    /,
) -> ClaudeCodeFailed:
    if deadline.expired:
        return ClaudeCodeTimedOut(timeout=launch.timeout, session_id=observed.session_id)

    if exit_code is None:  # the output was closed but the process kept running
        return ClaudeCodeUnfinished(
            stderr=stderr_tail.decode(errors="replace").strip(),
            session_id=observed.session_id,
        )

    # the error output has ended with the process, let the tail be complete
    try:
        await asyncio.wait_for(asyncio.shield(stderr_task), 1.0)

    except TimeoutError:
        pass  # held open by a child, what was read is what there is

    return ClaudeCodeEnded(
        exit_code=exit_code,
        stderr=stderr_tail.decode(errors="replace").strip(),
        session_id=observed.session_id,
    )


async def run_process(
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
        When the process timed out, ended without reporting a result or its output was not
        readable.
    """
    if sys.platform == "win32":
        raise ClaudeCodeUnavailable("Claude Code runs require a POSIX platform")

    process: asyncio.subprocess.Process = await _spawn(launch)
    stderr_tail: bytearray = bytearray()
    stderr_task: asyncio.Task[None] = asyncio.create_task(
        _drain(process, stderr_tail, launch.redactions)
    )
    exit_task: asyncio.Task[None] = asyncio.create_task(_end_output_after_exit(process))
    deadline: _Deadline = _Deadline(process, timeout=launch.timeout, grace=launch.grace)
    observed: _Observed = _Observed()
    try:
        await _send(process, launch.prompt)
        while line := await _read_line(process, observed):
            for event in parse_events(line):
                observed = observed.after(event)
                if isinstance(event, ClaudeCodeResult):
                    deadline.disarm()

                yield event

            if observed.result is not None:
                break  # nothing follows the result, children keeping the output open do not matter

        exit_code: int | None = await _exit_code(process, launch.grace)
        if observed.result is None:
            raise await _ended(
                launch,
                deadline,
                observed,
                exit_code,
                stderr_task,
                stderr_tail,
            )

    finally:
        deadline.release()
        await _finish(process, stderr_task, exit_task, launch.grace, launch.marker)


async def _drain(
    process: asyncio.subprocess.Process,
    tail: bytearray,
    secrets: Sequence[str],
    /,
) -> None:
    # an unread error output fills its pipe and blocks the process, only the tail is kept
    assert process.stderr is not None  # nosec: B101
    hidden: Sequence[bytes] = tuple(secret.encode() for secret in secrets)
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
    await _reap(marker)
    if process.stdin is not None and not process.stdin.is_closing():
        process.stdin.close()

    _close_output(process)
    stderr_task.cancel()
    exit_task.cancel()
    await asyncio.gather(stderr_task, exit_task, return_exceptions=True)
