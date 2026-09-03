from asyncio import Task, current_task, sleep
from typing import Any, cast

import pytest

from draive.mcp.client import MCPClient
from draive.resources import ResourcesRepository
from draive.tools import ToolsProvider


class _TaskRecordingSessionManager:
    """Records which task entered and exited the transport context manager.

    The MCP transports are built out of anyio task groups, which are task affine -
    entering and exiting one from different tasks raises at runtime.
    """

    def __init__(self) -> None:
        self.entered_in: Task[Any] | None = None
        self.exited_in: Task[Any] | None = None

    async def __aenter__(self) -> Any:
        self.entered_in = current_task()
        return _StubSession()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)
        self.exited_in = current_task()


class _StubSession:
    async def initialize(self) -> None:
        return None


@pytest.mark.asyncio
async def test_transport_is_entered_and_exited_within_the_same_task() -> None:
    manager = _TaskRecordingSessionManager()
    client = MCPClient(
        "test",
        session_manager=cast(Any, manager),
        features=(ResourcesRepository, ToolsProvider),
        tags=(),
    )

    _ = await client.__aenter__()
    entering_task: Task[Any] | None = current_task()
    await client.__aexit__(None, None, None)

    assert manager.entered_in is not None
    assert manager.exited_in is not None
    # both sides have to happen within one dedicated owner task, which is
    # deliberately not the task driving enter/exit
    assert manager.entered_in is manager.exited_in
    assert manager.entered_in is not entering_task


@pytest.mark.asyncio
async def test_transport_exit_completes_when_teardown_runs_in_another_task() -> None:
    """The context teardown does not generally run within the setup task."""
    manager = _TaskRecordingSessionManager()
    client = MCPClient(
        "test",
        session_manager=cast(Any, manager),
        features=(ResourcesRepository,),
        tags=(),
    )

    async def enter() -> None:
        _ = await client.__aenter__()

    async def exit_() -> None:
        await client.__aexit__(None, None, None)

    # separate tasks for setup and teardown - what previously raised
    # `Attempted to exit cancel scope in a different task than it was entered in`
    import asyncio

    await asyncio.create_task(enter())
    await sleep(0)
    await asyncio.create_task(exit_())

    assert manager.entered_in is manager.exited_in


class _FailingSessionManager:
    async def __aenter__(self) -> Any:
        raise RuntimeError("transport unavailable")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


@pytest.mark.asyncio
async def test_transport_failure_surfaces_from_enter() -> None:
    client = MCPClient(
        "test",
        session_manager=cast(Any, _FailingSessionManager()),
        features=(ResourcesRepository,),
        tags=(),
    )

    with pytest.raises(RuntimeError, match="transport unavailable"):
        _ = await client.__aenter__()
