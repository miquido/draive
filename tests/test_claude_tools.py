import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from pathlib import Path

import anyio
import pytest
from haiway import State, ctx

from draive import tool
from draive.claude import ClaudeCodeUnavailable
from draive.claude.tools import TOOLS_SERVER_KEY, TOOLS_SERVER_RULE, ClaudeCodeToolsBridge
from draive.mcp import MCPClient
from draive.tools import Toolbox


class _Marker(State):
    value: str


@tool(description="Read the marker of the agent.")
async def read_marker() -> str:
    return ctx.state(_Marker).value


@tool(description="Always fails.")
async def exploding() -> str:
    raise RuntimeError("tool exploded")


@tool(description="Returns the result to the user.", handling="output")
async def outputting() -> str:
    return "output"


async def _relays(
    directory: str,
    /,
) -> list[str]:
    process = await asyncio.create_subprocess_exec(
        "pgrep",
        "-f",
        directory,
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    return stdout.decode().split()


def test_server_constants() -> None:
    assert TOOLS_SERVER_KEY == "draive"
    assert TOOLS_SERVER_RULE == "mcp__draive"


def test_verify_accepts_response_tools() -> None:
    ClaudeCodeToolsBridge.verify(Toolbox.of(read_marker, exploding))


def test_verify_refuses_output_handling() -> None:
    with pytest.raises(ValueError, match="response"):
        ClaudeCodeToolsBridge.verify(Toolbox.of(read_marker, outputting))


@pytest.mark.parametrize(
    "name",
    (
        "a" * 116,  # with mcp__draive__ over 128 characters
        "with space",
        "dotted.name",
        "zażółć",
    ),
)
def test_verify_refuses_unsupported_names(name: str) -> None:
    with pytest.raises(ValueError, match="Tool name"):
        ClaudeCodeToolsBridge.verify(Toolbox.of(read_marker.updating(name=name)))


def test_verify_accepts_the_longest_name() -> None:
    ClaudeCodeToolsBridge.verify(Toolbox.of(read_marker.updating(name="a" * 115)))


@pytest.mark.asyncio
async def test_tools_are_served_within_the_context_of_the_scope() -> None:
    async with ctx.scope("test", _Marker(value="from the agent")):
        async with AsyncExitStack() as stack:
            bridge = await stack.enter_async_context(
                ClaudeCodeToolsBridge.serve(Toolbox.of(read_marker, exploding))
            )
            command = bridge.mcp_server["command"]
            args = bridge.mcp_server["args"]
            assert isinstance(command, str)
            assert isinstance(args, list)
            directory = os.path.dirname(str(args[-1]))
            assert os.path.isdir(directory)
            assert oct(os.stat(directory).st_mode & 0o777) == "0o700"

            clients = [
                MCPClient.stdio(
                    command=command,
                    args=[str(arg) for arg in args],
                    request_timeout=10,
                )
                for _ in range(2)
            ]
            fetched = []
            for client in clients:
                await stack.enter_async_context(client)
                fetched.append(await client.tools_fetch())

            for tools in fetched:
                assert {tool.name for tool in tools} == {"read_marker", "exploding"}

            results = await asyncio.gather(
                *(Toolbox.of(tools).call("read_marker", arguments={}) for tools in fetched)
            )
            assert all(result.to_str() == "from the agent" for result in results)

            assert await _relays(directory)  # relays are running

            # a failing tool does not break the server
            with pytest.raises(Exception, match="tool exploded"):
                await Toolbox.of(fetched[0]).call("exploding", arguments={})

            again = await Toolbox.of(fetched[0]).call("read_marker", arguments={})
            assert again.to_str() == "from the agent"

    assert not os.path.exists(directory)


@pytest.mark.asyncio
async def test_exit_removes_the_socket_and_stops_the_relays() -> None:
    async with ctx.scope("test", _Marker(value="x")):
        bridge_context = ClaudeCodeToolsBridge.serve(Toolbox.of(read_marker))
        bridge = await bridge_context.__aenter__()
        args = bridge.mcp_server["args"]
        assert isinstance(args, list)
        directory = os.path.dirname(str(args[-1]))
        command = bridge.mcp_server["command"]
        assert isinstance(command, str)
        async with MCPClient.stdio(
            command=command, args=[str(arg) for arg in args], request_timeout=10
        ) as _:
            assert await _relays(directory)
            await bridge_context.__aexit__(None, None, None)
            assert not os.path.exists(directory)
            for _ in range(40):
                if not await _relays(directory):
                    break

                await asyncio.sleep(0.05)

            assert not await _relays(directory)


@pytest.mark.asyncio
async def test_serve_refuses_too_long_socket_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    long_directory = tmp_path / ("d" * 120)
    long_directory.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(long_directory))
    with pytest.raises(ClaudeCodeUnavailable):
        async with ClaudeCodeToolsBridge.serve(Toolbox.of(read_marker)):
            pass


_blocked: dict[str, asyncio.Event | bool] = {}


@tool(description="Blocks until cancelled.")
async def blocking() -> str:
    started = _blocked["started"]
    assert isinstance(started, asyncio.Event)
    try:
        started.set()
        await asyncio.sleep(60)

    finally:
        _blocked["cancelled"] = True

    return "unreachable"


async def _hold_with_call_in_flight(
    holding: Callable[[asyncio.Event, list[ClaudeCodeToolsBridge]], Awaitable[None]],
    cancelling: Callable[[asyncio.Task[None]], None],
) -> tuple[ClaudeCodeToolsBridge, str]:
    _blocked.clear()
    _blocked["started"] = asyncio.Event()
    _blocked["cancelled"] = False
    ready = asyncio.Event()
    bridges: list[ClaudeCodeToolsBridge] = []
    holder = asyncio.create_task(holding(ready, bridges))
    async with ctx.scope("test", _Marker(value="x")):
        await asyncio.wait_for(ready.wait(), 10)
        bridge = bridges[0]
        command = bridge.mcp_server["command"]
        args = bridge.mcp_server["args"]
        assert isinstance(command, str)
        assert isinstance(args, list)
        directory = os.path.dirname(str(args[-1]))
        client = MCPClient.stdio(
            command=command, args=[str(arg) for arg in args], request_timeout=60
        )
        async with client:
            tools = await client.tools_fetch()
            call = asyncio.create_task(Toolbox.of(tools).call("blocking", arguments={}))
            await asyncio.wait_for(_started().wait(), 10)
            assert len(bridge._connections) == 1  # pyright: ignore[reportPrivateUsage]
            cancelling(holder)
            try:
                await asyncio.wait_for(asyncio.shield(holder), 10)

            except asyncio.CancelledError:
                pass

            # right after the bridge is gone, nothing is left behind
            assert not bridge._connections  # pyright: ignore[reportPrivateUsage]
            assert not os.path.exists(directory)
            assert _blocked["cancelled"] is True
            call.cancel()
            await asyncio.gather(call, return_exceptions=True)

    return bridge, directory


def _started() -> asyncio.Event:
    started = _blocked["started"]
    assert isinstance(started, asyncio.Event)
    return started


@pytest.mark.asyncio
async def test_cancelling_the_bridge_completes_the_teardown_with_a_call_in_flight() -> None:
    async def holding(ready: asyncio.Event, bridges: list[ClaudeCodeToolsBridge]) -> None:
        async with ClaudeCodeToolsBridge.serve(Toolbox.of(blocking)) as bridge:
            bridges.append(bridge)
            ready.set()
            await asyncio.sleep(60)

    await _hold_with_call_in_flight(holding, lambda holder: holder.cancel())


@pytest.mark.asyncio
async def test_an_anyio_scope_cancelling_the_bridge_completes_the_teardown() -> None:
    scope = anyio.CancelScope()

    async def holding(ready: asyncio.Event, bridges: list[ClaudeCodeToolsBridge]) -> None:
        with scope:
            async with ClaudeCodeToolsBridge.serve(Toolbox.of(blocking)) as bridge:
                bridges.append(bridge)
                ready.set()
                await anyio.sleep(60)

    await _hold_with_call_in_flight(holding, lambda _: scope.cancel())


class _Writer:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_a_connection_delivered_during_the_teardown_is_closed() -> None:
    async with ClaudeCodeToolsBridge.serve(Toolbox.of(read_marker)) as bridge:
        pass

    # the teardown has started, a connection accepted before it is delivered too late
    writer = _Writer()
    bridge._connected(None, None, writer, None)  # type: ignore[arg-type]

    assert writer.closed
    assert not bridge._connections
