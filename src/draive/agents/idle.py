from asyncio import AbstractEventLoop, Event, Handle, gather, get_running_loop
from types import TracebackType
from typing import Any, Self, final

__all__ = [
    "IdleMonitor",
]


@final
class IdleMonitor:
    def __init__(
        self,
        loop: AbstractEventLoop | None = None,
    ) -> None:
        self._loop: AbstractEventLoop = loop or get_running_loop()
        self._running_tasks: int = 0
        self._idle_status: Event = Event()
        # start as running but switches to idle soon if not entered any task after initializing
        self._scheduled_handle: Handle = self._loop.call_soon(self._idle_status.set)
        self._nested_statuses: list[IdleMonitor] = []

    def enter_task(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._scheduled_handle.cancel()
        self._idle_status.clear()
        self._running_tasks += 1

    def exit_task(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert self._running_tasks > 0, "Imbalanced RunningStatus task enter/exit"  # nosec: B101
        self._running_tasks -= 1
        if self._running_tasks > 0:
            return  # still running

        self._scheduled_handle.cancel()
        self._scheduled_handle = self._loop.call_soon(self._idle_status.set)

    def exit_all(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.idle:
            return  # already idle

        self._running_tasks = 0
        self._scheduled_handle.cancel()
        self._scheduled_handle = self._loop.call_soon(self._idle_status.set)

    def nested(self) -> Self:
        nested: Self = self.__class__()
        self._nested_statuses.append(nested)
        return nested

    def __enter__(self) -> None:
        self.enter_task()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.exit_task()

    @property
    def idle(self) -> bool:
        return self._idle_status.is_set() and all(status.idle for status in self._nested_statuses)

    async def wait_idle(self) -> None:
        while not self.idle:
            await gather(
                self._idle_status.wait(),
                *[status.wait_idle() for status in self._nested_statuses],
            )

        assert (  # nosec: B101
            self._running_tasks == 0
        ), "Invalid RunningStatus - notified idle with running tasks"
