from asyncio import Task, sleep

from draive.agents.idle import IdleMonitor
from pytest import mark


@mark.asyncio
async def test_not_idle_initially():
    monitor: IdleMonitor = IdleMonitor()
    assert not monitor.idle


@mark.asyncio
async def test_becomes_idle_initially():
    monitor: IdleMonitor = IdleMonitor()

    await sleep(0)  # allow loop to handle other tasks
    assert monitor.idle


@mark.asyncio
async def test_not_idle_when_entered_task():
    monitor: IdleMonitor = IdleMonitor()

    monitor.enter_task()

    await sleep(0)  # allow loop to handle other tasks
    assert not monitor.idle


@mark.asyncio
async def test_becomes_idle_after_exiting_task():
    monitor: IdleMonitor = IdleMonitor()

    monitor.enter_task()
    await sleep(0)  # allow loop to handle other tasks

    monitor.exit_task()
    assert not monitor.idle

    await sleep(0)  # allow loop to handle other tasks

    assert monitor.idle


@mark.asyncio
async def test_not_idle_when_nested_not_idle():
    monitor: IdleMonitor = IdleMonitor()
    nested: IdleMonitor = monitor.nested()

    nested.enter_task()

    await sleep(0)  # allow loop to handle other tasks
    assert not monitor.idle


@mark.asyncio
async def test_becomes_idle_after_nested_becomes_idle():
    monitor: IdleMonitor = IdleMonitor()
    nested: IdleMonitor = monitor.nested()

    nested.enter_task()
    await sleep(0)  # allow loop to handle other tasks

    nested.exit_task()
    assert not monitor.idle

    await sleep(0)  # allow loop to handle other tasks

    assert monitor.idle


@mark.asyncio
async def test_waits_for_being_idle():
    monitor: IdleMonitor = IdleMonitor()
    monitor.enter_task()

    async def exit_task() -> None:
        await sleep(0)  # allow loop to handle other tasks
        monitor.exit_task()

    Task(exit_task())

    assert not monitor.idle
    await monitor.wait_idle()
    assert monitor.idle


@mark.asyncio
async def test_waits_for_being_idle_with_nested():
    monitor: IdleMonitor = IdleMonitor()
    nested: IdleMonitor = monitor.nested()
    nested.enter_task()

    async def exit_task() -> None:
        await sleep(0)  # allow loop to handle other tasks
        nested.exit_task()

    Task(exit_task())

    assert not monitor.idle
    await monitor.wait_idle()
    assert monitor.idle


@mark.asyncio
async def test_resets_when_entering_task():
    monitor: IdleMonitor = IdleMonitor()

    await sleep(0)  # allow loop to handle other tasks
    assert monitor.idle

    monitor.enter_task()
    await sleep(0)  # allow loop to handle other tasks
    assert not monitor.idle


@mark.asyncio
async def test_resets_when_nested_entering_task():
    monitor: IdleMonitor = IdleMonitor()
    nested: IdleMonitor = monitor.nested()

    await sleep(0)  # allow loop to handle other tasks
    assert monitor.idle

    nested.enter_task()
    await sleep(0)  # allow loop to handle other tasks
    assert not monitor.idle
