import asyncio
import contextvars
import os
import re
import shutil
import sys
import tempfile
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Final, Self, final

from haiway import BasicValue, ctx

from draive.claude._completion import completed
from draive.claude.exceptions import ClaudeCodeUnavailable
from draive.tools import Toolbox

if TYPE_CHECKING:
    from draive.mcp import MCPServer

__all__ = (
    "TOOLS_SERVER_KEY",
    "TOOLS_SERVER_RULE",
    "ClaudeCodeToolsBridge",
)

TOOLS_SERVER_KEY: Final[str] = "draive"
TOOLS_SERVER_RULE: Final[str] = f"mcp__{TOOLS_SERVER_KEY}"

_TOOL_PREFIX: Final[str] = f"{TOOLS_SERVER_RULE}__"
_NAME_LIMIT: Final[int] = 128  # Claude Code silently drops the tools with longer names
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]+")
_SOCKET_LIMIT: Final[int] = 100  # AF_UNIX paths are limited to ~104 bytes on macOS
_LINE_LIMIT: Final[int] = 64 * 1024 * 1024  # tool results can be big
_CLOSE_LIMIT: Final[float] = 1.0  # flushing a closed connection, aborted after it
_RELAY: Final[str] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mcp_relay.py")


@final
class ClaudeCodeToolsBridge:
    """Serves draive tools to a single Claude Code run as an MCP server.

    Claude Code starts stdio MCP servers as its own child processes, while the tools live in
    the process of the agent. The bridge listens on a private Unix socket and a small relay
    script, started by Claude Code, connects its stdio to it.
    """

    __slots__ = ("_connections", "_directory", "_socket", "_stopping", "_writers")

    def __init__(
        self,
        directory: str,
        /,
    ) -> None:
        self._directory: Final[str] = directory
        self._socket: Final[str] = os.path.join(directory, "mcp.sock")
        self._connections: Final[set[asyncio.Task[None]]] = set()
        self._stopping: bool = False
        # writers of the accepted connections, also of handlers which never started
        self._writers: Final[set[asyncio.StreamWriter]] = set()

    @staticmethod
    def verify(
        toolbox: Toolbox,
        /,
    ) -> None:
        """Verify that all tools can be served to Claude Code.

        Parameters
        ----------
        toolbox
            Tools to be served.

        Raises
        ------
        ValueError
            When a tool does not return its results to the model, or its name is not accepted
            by Claude Code.
        """
        for name, tool in toolbox.tools.items():
            if tool.handling != "response":
                raise ValueError(
                    f"Tool '{name}' of Claude Code uses '{tool.handling}' handling, only tools"
                    " with 'response' handling are supported - only their results are returned"
                    " to the model"
                )

            if _NAME_PATTERN.fullmatch(name) is None:
                raise ValueError(
                    f"Tool name '{name}' is not supported by Claude Code,"
                    " only letters, digits, '_' and '-' are allowed"
                )

            if len(_TOOL_PREFIX) + len(name) > _NAME_LIMIT:
                raise ValueError(
                    f"Tool name '{name}' is too long for Claude Code, at most"
                    f" {_NAME_LIMIT - len(_TOOL_PREFIX)} characters are allowed"
                )

    @classmethod
    @asynccontextmanager
    async def serve(
        cls,
        toolbox: Toolbox,
        /,
    ) -> AsyncGenerator[Self]:
        """Serve tools until the context exits, within the current context.

        Parameters
        ----------
        toolbox
            Tools to be served.

        Raises
        ------
        ClaudeCodeUnavailable
            When the ``mcp`` extra is not installed or the socket path is too long.
        """
        try:
            from draive.mcp import MCPServer

        except ImportError as exc:
            raise ClaudeCodeUnavailable(
                "draive tools of Claude Code need the mcp extra, install draive[mcp]"
            ) from exc

        server: MCPServer = MCPServer(name=TOOLS_SERVER_KEY, tools=toolbox)
        directory: str = tempfile.mkdtemp(prefix="draive-mcp-")  # owned from here on
        bridge: Self = cls(directory)
        listening: asyncio.Server | None = None
        try:
            if len(os.fsencode(bridge._socket)) > _SOCKET_LIMIT:
                raise ClaudeCodeUnavailable(
                    f"the socket path of draive tools is too long: {bridge._socket}"
                )

            # tools see the state of the agent - every connection runs within a copy of the
            # context entering the bridge, not whichever asyncio gives its callbacks
            context: contextvars.Context = contextvars.copy_context()
            listening = await asyncio.start_unix_server(
                lambda reader, writer: bridge._connected(server, reader, writer, context),
                path=bridge._socket,
                limit=_LINE_LIMIT,
            )
            yield bridge

        finally:
            await completed(asyncio.create_task(bridge._teardown(listening)))

    @property
    def mcp_server(self) -> Mapping[str, BasicValue]:
        """MCP server entry of Claude Code, a relay process connected to the bridge."""
        return {
            "command": sys.executable,
            "args": ["-I", _RELAY, self._socket],
        }

    async def _teardown(
        self,
        listening: asyncio.Server | None,
        /,
    ) -> None:
        try:
            # a connection accepted meanwhile is delivered later, it is closed instead of served
            self._stopping = True
            if listening is not None:
                listening.close()

            connections: list[asyncio.Task[None]] = list(self._connections)
            for connection in connections:
                connection.cancel()

            # the transports are owned here, a handler cancelled before its first step never
            # closes its own and the listener would wait for it
            for writer in tuple(self._writers):
                writer.transport.abort()

            await asyncio.gather(*connections, return_exceptions=True)
            if listening is not None:
                with suppress(Exception):
                    await asyncio.wait_for(listening.wait_closed(), _CLOSE_LIMIT)

        finally:
            await asyncio.to_thread(shutil.rmtree, self._directory, True)

    def _connected(
        self,
        server: MCPServer,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        context: contextvars.Context,
        /,
    ) -> None:
        if self._stopping:
            writer.close()
            return

        self._writers.add(writer)
        task: asyncio.Task[None] = asyncio.create_task(
            self._handle(server, reader, writer),
            context=context.copy(),
        )
        self._connections.add(task)
        task.add_done_callback(self._connections.discard)

    async def _handle(
        self,
        server: MCPServer,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        /,
    ) -> None:
        try:
            await _serve_connection(server, reader, writer)

        except* ConnectionError:
            pass  # the relay ended with its run, nothing waits for the answers

        except* Exception as exc:
            ctx.log_error("Claude Code tools connection failed", exception=exc)

        finally:
            self._writers.discard(writer)
            writer.close()
            try:
                # a relay which stopped reading would hold the flush, and the teardown, forever
                await asyncio.wait_for(writer.wait_closed(), _CLOSE_LIMIT)

            except Exception:
                writer.transport.abort()


async def _serve_connection(
    server: MCPServer,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    /,
) -> None:
    # the transport of mcp.server.stdio.stdio_server over a socket
    import anyio
    import anyio.lowlevel
    import mcp_types as types
    from mcp.shared.message import SessionMessage

    # public memory streams, as the stdio client of mcp uses - tools run within the context of
    # the connection, which is the one of the agent
    read_stream_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](
        0
    )
    write_stream, write_stream_reader = anyio.create_memory_object_stream[SessionMessage](0)

    async def inbound() -> None:
        try:
            async with read_stream_writer:
                async for line in reader:
                    try:
                        message = types.jsonrpc_message_adapter.validate_json(line, by_name=False)

                    except Exception as exc:
                        await read_stream_writer.send(exc)
                        continue

                    await read_stream_writer.send(SessionMessage(message))

        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def outbound() -> None:
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    writer.write(
                        session_message.message.model_dump_json(
                            by_alias=True,
                            exclude_unset=True,
                        ).encode()
                        + b"\n"
                    )
                    await writer.drain()

        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(inbound)
        tasks.start_soon(outbound)
        await server.run(read_stream, write_stream)
        tasks.cancel_scope.cancel()
