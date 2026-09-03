import json
from asyncio import AbstractEventLoop, Event, Future, Task, gather, get_running_loop
from base64 import b64decode, urlsafe_b64decode, urlsafe_b64encode
from collections.abc import AsyncGenerator, Callable, Collection, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from itertools import chain
from types import TracebackType
from typing import Any, Self, cast, final
from urllib.parse import ParseResult, urlparse, urlunparse
from uuid import uuid4

from haiway import (
    BasicValue,
    Meta,
    MetaTags,
    Paginated,
    Pagination,
    PaginationToken,
    as_dict,
    as_list,
    as_tuple,
    ctx,
)
from httpx2 import AsyncClient, Timeout
from mcp import ClientSession, ListToolsResult, StdioServerParameters, stdio_client
from mcp import Tool as MCPTool
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import AudioContent as MCPAudioContent
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ListResourcesResult,
    PaginatedRequestParams,
    ReadResourceResult,
    TextResourceContents,
)
from mcp.types import EmbeddedResource as MCPEmbeddedResource
from mcp.types import ImageContent as MCPImageContent
from mcp.types import Resource as MCPResource
from mcp.types import ResourceLink as MCPResourceLink
from mcp.types import TextContent as MCPTextContent

from draive.models import ModelToolParametersSpecification
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.resources import ResourceContent, ResourceReference, ResourcesRepository
from draive.tools import CoroutineTool, Tool, ToolException, ToolsProvider

__all__ = (
    "MCPClient",
    "MCPClients",
)

DEFAULT_PAGINATION_LIMIT = 32
_LOCAL_PAGINATION_TOKEN_PREFIX = "mcp:"  # nosec B105


def _encode_pagination_token(
    data: Mapping[str, Any],
) -> str:
    encoded: str = urlsafe_b64encode(json.dumps(data).encode()).decode()
    return f"{_LOCAL_PAGINATION_TOKEN_PREFIX}{encoded}"


def _decode_pagination_token(
    token: PaginationToken | None,
) -> dict[str, Any] | None:
    if not isinstance(token, str) or not token.startswith(_LOCAL_PAGINATION_TOKEN_PREFIX):
        return None

    try:
        encoded: str = token.removeprefix(_LOCAL_PAGINATION_TOKEN_PREFIX)
        decoded: Any = json.loads(urlsafe_b64decode(encoded.encode()).decode())
        if isinstance(decoded, dict):
            return cast(dict[str, Any], decoded)

        else:
            return None

    except Exception:
        return None


def _decode_single_page_cursor(
    token: PaginationToken | None,
) -> tuple[str | None, int]:
    cursor: str | None = token if isinstance(token, str) else None
    offset: int = 0
    decoded = _decode_pagination_token(token)
    if decoded is None or decoded.get("kind") != "single":
        return cursor, offset

    token_value: Any = decoded.get("cursor")
    if isinstance(token_value, str | None):
        cursor = token_value

    offset_value: Any = decoded.get("offset")
    if isinstance(offset_value, int):
        offset = max(offset_value, 0)

    return cursor, offset


def _decode_aggregate_state(
    *,
    token: PaginationToken | None,
    server_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {
        server_id: {
            "cursor": None,
            "offset": 0,
            "done": False,
        }
        for server_id in server_ids
    }
    decoded = _decode_pagination_token(token)
    if decoded is None or decoded.get("kind") != "aggregate":
        return state

    state_value: Any = decoded.get("state")
    if not isinstance(state_value, Mapping):
        return state
    state_mapping: Mapping[str, Any] = cast(Mapping[str, Any], state_value)

    for server_id in server_ids:
        server_state: Any = state_mapping.get(server_id)
        if not isinstance(server_state, Mapping):
            continue

        server_state_mapping: Mapping[str, Any] = cast(Mapping[str, Any], server_state)

        cursor_value: Any = server_state_mapping.get("cursor")
        if isinstance(cursor_value, str | None):
            state[server_id]["cursor"] = cursor_value

        # offset is absent in tokens produced before it was introduced
        offset_value: Any = server_state_mapping.get("offset")
        if isinstance(offset_value, int):
            state[server_id]["offset"] = max(offset_value, 0)

        done_value: Any = server_state_mapping.get("done")
        if isinstance(done_value, bool):
            state[server_id]["done"] = done_value

    return state


@final
class MCPClient:
    @classmethod
    def stdio(
        cls,
        identifier: str | None = None,
        *,
        command: str,
        args: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
        features: Collection[type[ResourcesRepository] | type[ToolsProvider]] | None = None,
        tags: MetaTags | None = None,
    ) -> Self:
        @asynccontextmanager
        async def mcp_stdio_session() -> AsyncGenerator[ClientSession]:
            async with stdio_client(
                StdioServerParameters(
                    command=command,
                    args=as_list(args) if args is not None else [],
                    env=as_dict(env),
                )
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session

        return cls(
            identifier or str(uuid4()),
            session_manager=mcp_stdio_session(),
            features=features if features is not None else (ResourcesRepository, ToolsProvider),
            tags=tags if tags is not None else (),
        )

    @classmethod
    def sse(
        cls,
        identifier: str | None = None,
        *,
        url: str,
        headers: Mapping[str, Any] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
        features: Collection[type[ResourcesRepository] | type[ToolsProvider]] | None = None,
        tags: MetaTags | None = None,
    ) -> Self:
        @asynccontextmanager
        async def mcp_sse_session() -> AsyncGenerator[ClientSession]:
            async with sse_client(
                url=url,
                headers=as_dict(headers),
                timeout=timeout,
                sse_read_timeout=sse_read_timeout,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session

        return cls(
            identifier or str(uuid4()),
            session_manager=mcp_sse_session(),
            features=features if features is not None else (ResourcesRepository, ToolsProvider),
            tags=tags if tags is not None else (),
        )

    @classmethod
    def streamable_http(
        cls,
        identifier: str | None = None,
        *,
        url: str,
        headers: Mapping[str, Any] | None = None,
        timeout: float = 30,
        sse_read_timeout: float = 60 * 5,
        terminate_on_close: bool = True,
        features: Collection[type[ResourcesRepository] | type[ToolsProvider]] | None = None,
        tags: MetaTags | None = None,
    ) -> Self:
        """Prepare a client using the Streamable HTTP transport.

        This is the current remote transport, `sse` is deprecated by the protocol.
        """

        @asynccontextmanager
        async def mcp_streamable_http_session() -> AsyncGenerator[ClientSession]:
            async with AsyncClient(
                # part of the recommended MCP client defaults
                follow_redirects=True,
                headers=as_dict(headers),
                timeout=Timeout(timeout, read=sse_read_timeout),
            ) as http_client:
                async with streamable_http_client(
                    url,
                    http_client=http_client,
                    terminate_on_close=terminate_on_close,
                ) as (read, write):
                    async with ClientSession(read, write) as session:
                        yield session

        return cls(
            identifier or str(uuid4()),
            session_manager=mcp_streamable_http_session(),
            features=features if features is not None else (ResourcesRepository, ToolsProvider),
            tags=tags if tags is not None else (),
        )

    __slots__ = (
        "_features",
        "_session",
        "_session_closing",
        "_session_manager",
        "_session_owner",
        "identifier",
        "tags",
    )

    def __init__(
        self,
        identifier: str,
        *,
        session_manager: AbstractAsyncContextManager[ClientSession],
        features: Collection[type[ResourcesRepository] | type[ToolsProvider]],
        tags: MetaTags,
    ) -> None:
        self.identifier: str = identifier
        self._session_manager: AbstractAsyncContextManager[ClientSession] = session_manager
        self._session: ClientSession
        self._session_owner: Task[None]  # created on enter
        self._session_closing: Event  # created on enter
        self._features: Collection[type[ResourcesRepository] | type[ToolsProvider]] = features
        self.tags: MetaTags = tags

    def _meta(
        self,
        *,
        include_tags: bool = True,
        **values: BasicValue,
    ) -> Meta:
        base: dict[str, BasicValue] = {
            "mcp_server": self.identifier,
        }
        if include_tags:
            base["tags"] = tuple(self.tags)

        if values:
            base.update(values)

        return Meta.of(base)

    async def resources_list(
        self,
        pagination: Pagination | None,
        **extra: Any,
    ) -> Paginated[ResourceReference]:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized through async context entering"

        pagination = (
            pagination
            if pagination is not None
            else Pagination.of(
                token=None,
                limit=DEFAULT_PAGINATION_LIMIT,
            )
        )
        if pagination.limit <= 0:
            return Paginated[ResourceReference].of(
                (),
                pagination=pagination.with_token(None),
            )

        starting_cursor, page_offset = _decode_single_page_cursor(pagination.token)
        remaining: int = pagination.limit
        references: list[ResourceReference] = []
        cursor: str | None = starting_cursor
        offset: int = page_offset
        next_token: str | None = None
        while remaining > 0:
            request_params: PaginatedRequestParams | None
            if cursor is not None:
                request_params = PaginatedRequestParams(cursor=cursor)

            else:
                request_params = None

            result: ListResourcesResult = await self._session.list_resources(
                params=request_params,
            )
            current_resources = result.resources

            if offset >= len(current_resources):
                # Keep walking pages when stale/local offset points past current data.
                offset -= len(current_resources)
                if result.next_cursor is None or result.next_cursor == cursor:
                    next_token = None
                    break

                cursor = result.next_cursor
                continue

            available = current_resources[offset:]
            consumed: int = min(len(available), remaining)
            references.extend(
                self._resource_reference(resource) for resource in available[:consumed]
            )
            remaining -= consumed
            if consumed < len(available):
                next_token = _encode_pagination_token(
                    {
                        "kind": "single",
                        "cursor": cursor,
                        "offset": offset + consumed,
                    }
                )
                break

            if result.next_cursor is None or result.next_cursor == cursor:
                next_token = None
                break

            cursor = result.next_cursor
            offset = 0
            next_token = cursor

        return Paginated[ResourceReference].of(
            tuple(references),
            pagination=pagination.with_token(next_token),
        )

    def _resource_reference(
        self,
        resource: MCPResource,
    ) -> ResourceReference:
        return ResourceReference(
            # uri is a plain str since mcp 2.0
            uri=self._with_uri_identifier(resource.uri),
            mime_type=resource.mime_type
            if resource.mime_type is not None
            else "application/octet-stream",
            meta=self._meta(
                name=resource.name,
                description=resource.description,
            ),
        )

    async def resource_fetch(
        self,
        uri: str,
        **extra: Any,
    ) -> Collection[ResourceReference] | ResourceContent | None:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized through async context entering"

        result: ReadResourceResult = await self._session.read_resource(
            uri=self._without_uri_identifier(uri)
        )

        match result.contents:
            case [resource]:
                # if there is only a single element return it directly
                return self._convert_resource_content(resource)

            case [*resources]:
                # otherwise convert to references ignoring content
                return [
                    ResourceReference(
                        # the identifier is required to resolve the reference back to this client
                        uri=self._with_uri_identifier(resource.uri),
                        mime_type=resource.mime_type
                        if resource.mime_type is not None
                        else "application/octet-stream",
                        meta=self._meta(),
                    )
                    for resource in resources
                ]

    async def resource_upload(
        self,
        uri: str,
        content: ResourceContent,
        **extra: Any,
    ) -> Meta:
        raise NotImplementedError("Resource uploading is not supported by MCP servers")

    async def resource_delete(
        self,
        uri: str,
        **extra: Any,
    ) -> None:
        raise NotImplementedError("Resource deleting is not supported by MCP servers")

    async def tools_fetch(
        self,
        **extra: Any,
    ) -> Sequence[Tool]:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized through async context entering"

        mcp_tools: list[MCPTool] = []
        cursor: str | None = None
        while True:
            request_params: PaginatedRequestParams | None
            if cursor is not None:
                request_params = PaginatedRequestParams(cursor=cursor)

            else:
                request_params = None

            result: ListToolsResult = await self._session.list_tools(params=request_params)
            mcp_tools.extend(result.tools)
            if result.next_cursor is None or result.next_cursor == cursor:
                break

            cursor = result.next_cursor

        return tuple(
            _convert_tool(
                tool,
                tool_call=self._tool_call,
                source=self.identifier,
                tags=self.tags,
            )
            for tool in mcp_tools
        )

    async def _tool_call(
        self,
        name: str,
        arguments: Mapping[str, BasicValue],
    ) -> MultimodalContent:
        result: CallToolResult = await self._session.call_tool(
            name=name,
            arguments=as_dict(arguments),
        )

        if result.is_error:
            # tool errors are reported within the result to let the model correct its call,
            # content is converted defensively to not hide the failure behind a conversion error
            error_content: MultimodalContent = MultimodalContent.of(
                *await gather(*(_convert_error_content(part) for part in result.content))
            )
            raise ToolException(
                f"Remote tool {name} failed: {error_content.to_str()}",
                tool=name,
                meta=self._meta(),
            )

        return MultimodalContent.of(
            *await gather(*(_convert_content(part) for part in result.content))
        )

    def _with_uri_identifier(
        self,
        uri: str,
        /,
    ) -> str:
        if not uri:
            return uri

        parsed: ParseResult = urlparse(uri)
        if parsed.netloc:
            return urlunparse(
                (
                    parsed.scheme,
                    f"{self.identifier}.{parsed.netloc}",
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )

        else:
            # Ensure path starts with /
            path: str = parsed.path
            if not path.startswith("/"):
                path = "/" + path

            return urlunparse(
                (
                    # default to mcp scheme if empty
                    parsed.scheme or "mcp",
                    # use identifier as domain/netloc
                    self.identifier,
                    path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )

    def _without_uri_identifier(
        self,
        uri: str,
        /,
    ) -> str:
        if not uri:
            return uri

        parsed: ParseResult = urlparse(uri)
        if not parsed.netloc:
            return uri

        if parsed.netloc == self.identifier:
            return urlunparse(
                (
                    parsed.scheme if parsed.scheme != "mcp" else "",
                    "",
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )

        match parsed.netloc.split(".", 1):
            case [identifier, netloc] if identifier == self.identifier:
                return urlunparse(
                    (
                        parsed.scheme,
                        netloc,
                        parsed.path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment,
                    )
                )

            case _:
                return uri

    def _convert_resource_content(
        self,
        resource: TextResourceContents | BlobResourceContents,
        /,
    ) -> ResourceContent:
        match resource:
            case TextResourceContents() as text_resource:
                # `text` carries the already decoded content, it has to be encoded back
                return ResourceContent.of(
                    text_resource.text.encode(),
                    mime_type=text_resource.mime_type or "text/plain",
                    meta=self._meta(),
                )

            case BlobResourceContents() as blob_resource:
                return ResourceContent(
                    data=blob_resource.blob,
                    mime_type=blob_resource.mime_type or "application/octet-stream",
                    meta=self._meta(),
                )

    async def __aenter__(self) -> Sequence[ResourcesRepository | ToolsProvider]:
        # the transports are built out of anyio task groups, which are task affine -
        # entering and exiting one from different tasks raises, and context teardown
        # does not generally run within the task which performed the setup.
        # a dedicated task owns the whole lifetime of the session instead.
        loop: AbstractEventLoop = get_running_loop()
        prepared: Future[ClientSession] = loop.create_future()
        self._session_closing = Event()

        async def session_owner() -> None:
            try:
                async with self._session_manager as session:
                    prepared.set_result(session)
                    await self._session_closing.wait()

            except BaseException as exc:
                if prepared.done():
                    raise  # surfaces when awaiting the task on exit

                prepared.set_exception(exc)

        self._session_owner = loop.create_task(session_owner())
        self._session = await prepared
        try:
            await self._session.initialize()

        except BaseException:
            await self._close_session()
            raise

        features: list[ResourcesRepository | ToolsProvider] = []
        if ResourcesRepository in self._features:
            features.append(
                ResourcesRepository(
                    list_fetching=self.resources_list,
                    fetching=self.resource_fetch,
                    uploading=self.resource_upload,
                    deleting=self.resource_delete,
                    meta=self._meta(),
                )
            )

        if ToolsProvider in self._features:
            features.append(
                ToolsProvider(
                    loading=self.tools_fetch,
                    meta=self._meta(),
                )
            )

        return features

    async def _close_session(self) -> None:
        self._session_closing.set()
        try:
            # the owner task performs the actual transport teardown
            await self._session_owner

        finally:
            del self._session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # the transport is closed regularly instead of being thrown into - it only
        # has to release the connection, while the cause propagates through the scope
        await self._close_session()


async def _convert_content(  # noqa: C901, PLR0911
    content: MCPTextContent
    | MCPImageContent
    | MCPAudioContent
    | MCPResourceLink
    | MCPEmbeddedResource,
    /,
) -> MultimodalContent:
    match content:
        case MCPTextContent() as text:
            return MultimodalContent.of(text.text)

        case MCPImageContent() as image:
            return MultimodalContent.of(
                ResourceContent.of(
                    b64decode(image.data),
                    mime_type=image.mime_type,
                )
            )

        case MCPAudioContent() as audio:
            return MultimodalContent.of(
                ResourceContent.of(
                    b64decode(audio.data),
                    mime_type=audio.mime_type,
                )
            )

        case MCPEmbeddedResource() as resource:
            match resource.resource:
                case TextResourceContents() as text:
                    return MultimodalContent.of(TextContent(text=text.text))

                case BlobResourceContents() as blob:
                    match blob.mime_type:
                        case None:
                            raise NotImplementedError(
                                "Unsupported embedded resource - missing mime!"
                            )

                        case "text/plain":
                            return MultimodalContent.of(
                                TextContent(text=b64decode(blob.blob).decode())
                            )

                        case "application/json":
                            return MultimodalContent.of(
                                ArtifactContent.of(
                                    json.loads(b64decode(blob.blob)),
                                    category="json",
                                )
                            )

                        case other:
                            # try to match supported media or raise an exception
                            return MultimodalContent.of(
                                ResourceContent.of(
                                    b64decode(blob.blob),
                                    mime_type=other,
                                )
                            )

        case MCPResourceLink():
            raise NotImplementedError("MCP resource links are not supported yet")


async def _convert_error_content(
    content: MCPTextContent
    | MCPImageContent
    | MCPAudioContent
    | MCPResourceLink
    | MCPEmbeddedResource,
    /,
) -> MultimodalContent:
    """Convert error result content without ever raising.

    Unsupported parts are replaced with a placeholder to not shadow the actual failure.
    """

    try:
        return await _convert_content(content)

    except Exception as exc:
        ctx.log_warning(
            f"Unsupported MCP tool error content ({content.type})",
            exception=exc,
        )
        return MultimodalContent.of(f"<{content.type}/>")


def _convert_tool(
    mcp_tool: MCPTool,
    /,
    *,
    tool_call: Callable[[str, Mapping[str, BasicValue]], Coroutine[None, None, MultimodalContent]],
    source: str,
    tags: MetaTags,
) -> Tool:
    name: str = mcp_tool.name

    async def remote_call(**arguments: Any) -> MultimodalContent:
        return await tool_call(
            name,
            arguments,
        )

    return CoroutineTool(
        name=name,
        description=mcp_tool.description,
        parameters=cast(
            ModelToolParametersSpecification,
            {
                **mcp_tool.input_schema,
                "additionalProperties": False,
            },
        ),
        function=remote_call,
        handling="response",
        meta=Meta.of(
            {
                "mcp_server": source,
                "tags": as_tuple(tags),
            }
        ),
    )


@final
class MCPClients:
    __slots__ = (
        "_clients",
        "_prompts",
        "_resources",
        "_tools",
    )

    def __init__(
        self,
        client: MCPClient,
        *clients: MCPClient,
    ) -> None:
        self._clients: Mapping[str, MCPClient] = {c.identifier: c for c in [client, *clients]}
        self._resources: Mapping[str, ResourcesRepository]
        self._tools: Mapping[str, ToolsProvider]

    async def resources_list(
        self,
        *,
        mcp_server: str | None = None,
        pagination: Pagination | None,
        **extra: Any,
    ) -> Paginated[ResourceReference]:
        pagination = (
            pagination
            if pagination is not None
            else Pagination.of(
                token=None,
                limit=DEFAULT_PAGINATION_LIMIT,
            )
        )
        if mcp_server is None:
            return await self._resources_list_aggregate(
                pagination=pagination,
                **extra,
            )

        elif resources := self._resources.get(mcp_server):
            return await resources.fetch_list(
                pagination=pagination,
                **extra,
            )

        else:
            return Paginated[ResourceReference].of(
                (),
                pagination=pagination.with_token(None),
            )

    async def _resources_list_aggregate(
        self,
        *,
        pagination: Pagination,
        **extra: Any,
    ) -> Paginated[ResourceReference]:
        server_ids: tuple[str, ...] = tuple(self._resources.keys())
        aggregate_state: dict[str, dict[str, Any]] = _decode_aggregate_state(
            token=pagination.token,
            server_ids=server_ids,
        )
        references: list[ResourceReference] = []
        remaining: int = pagination.limit
        while remaining > 0:
            pending_ids: list[str] = [
                server_id
                for server_id in server_ids
                if not bool(aggregate_state[server_id]["done"])
            ]
            if not pending_ids:
                break

            pages = await gather(
                *[
                    self._resources[server_id].fetch_list(
                        pagination=Pagination.of(
                            token=cast(str | None, aggregate_state[server_id]["cursor"]),
                            # already delivered items are requested again to be skipped below
                            limit=cast(int, aggregate_state[server_id]["offset"]) + remaining,
                        ),
                        **extra,
                    )
                    for server_id in pending_ids
                ]
            )

            progress_made: bool = False
            for server_id, page in zip(pending_ids, pages, strict=True):
                if remaining <= 0:
                    # remaining pages are discarded without advancing their cursors
                    break

                state: dict[str, Any] = aggregate_state[server_id]
                current_cursor: str | None = cast(str | None, state["cursor"])
                offset: int = cast(int, state["offset"])
                available: Sequence[ResourceReference] = tuple(page.items)[offset:]
                consumed: int = min(len(available), remaining)
                if consumed > 0:
                    references.extend(available[:consumed])
                    remaining -= consumed
                    progress_made = True

                if consumed < len(available):
                    # keep the overflow reachable through the same cursor with a larger offset
                    state["offset"] = offset + consumed
                    continue

                page_token: PaginationToken | None = page.pagination.token
                # the state is serialized into the token, only plain strings can be carried over
                next_cursor: str | None = str(page_token) if page_token is not None else None
                state["offset"] = 0
                if next_cursor is None or next_cursor == current_cursor:
                    # a repeated cursor means the source can't advance any further
                    state["cursor"] = None
                    state["done"] = True

                else:
                    state["cursor"] = next_cursor
                    state["done"] = False
                    progress_made = True

            if not progress_made:
                break

        next_token: str | None = None
        if any(not bool(state["done"]) for state in aggregate_state.values()):
            next_token = _encode_pagination_token(
                {
                    "kind": "aggregate",
                    "state": aggregate_state,
                }
            )

        return Paginated[ResourceReference].of(
            tuple(references[: pagination.limit]),
            pagination=pagination.with_token(next_token),
        )

    async def resource_fetch(
        self,
        uri: str,
        **extra: Any,
    ) -> Collection[ResourceReference] | ResourceContent | None:
        if client_identifier := self._client_identifier_for_uri(uri):
            return await self._resources[client_identifier].fetching(uri)

        else:
            ctx.log_warning(f"Requested resource ({uri}) from unknown source")
            return None

    async def resource_upload(
        self,
        uri: str,
        content: ResourceContent,
        **extra: Any,
    ) -> Meta:
        raise NotImplementedError("Resource uploading is not supported by MCP servers")

    async def resource_delete(
        self,
        uri: str,
        **extra: Any,
    ) -> None:
        raise NotImplementedError("Resource deleting is not supported by MCP servers")

    def _client_identifier_for_uri(
        self,
        uri: str,
        /,
    ) -> str | None:
        """Find server associated with URI."""
        if not uri:
            return None

        parsed: ParseResult = urlparse(uri)
        if not parsed.netloc:
            return None

        if parsed.netloc in self._clients.keys():
            return parsed.netloc

        match parsed.netloc.split(".", 1):
            case [identifier, _]:
                return identifier

            case _:
                return None

    async def tools_fetch(
        self,
        *,
        mcp_server: str | None = None,
        **extra: Any,
    ) -> Sequence[Tool]:
        if mcp_server is None:
            return tuple(
                chain.from_iterable(
                    await gather(*[client.load(**extra) for client in self._tools.values()])
                )
            )

        elif tools := self._tools.get(mcp_server):
            return await tools.load(**extra)

        else:
            return ()

    async def __aenter__(self) -> Sequence[ResourcesRepository | ToolsProvider]:
        features: Sequence[Sequence[ResourcesRepository | ToolsProvider]] = await gather(
            *[client.__aenter__() for client in self._clients.values()]
        )

        self._resources = {}
        self._prompts = {}
        self._tools = {}

        for states in features:
            for state in states:
                if isinstance(state, ResourcesRepository):
                    self._resources[cast(str, state.meta["mcp_server"])] = state

                if isinstance(state, ToolsProvider):
                    self._tools[cast(str, state.meta["mcp_server"])] = state

        inherited_features: list[ResourcesRepository | ToolsProvider] = []
        if self._resources:
            inherited_features.append(
                ResourcesRepository(
                    list_fetching=self.resources_list,
                    fetching=self.resource_fetch,
                    uploading=self.resource_upload,
                    deleting=self.resource_delete,
                    meta=Meta.of({"mcp_server": "mcp_aggregate"}),
                )
            )

        if self._tools:
            inherited_features.append(
                ToolsProvider(
                    loading=self.tools_fetch,
                    meta=Meta.of({"mcp_server": "mcp_aggregate"}),
                )
            )

        return inherited_features

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await gather(
            *[
                client.__aexit__(
                    exc_type,
                    exc_val,
                    exc_tb,
                )
                for client in self._clients.values()
            ]
        )
