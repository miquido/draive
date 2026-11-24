import json
import re
from asyncio import (
    FIRST_EXCEPTION,
    AbstractEventLoop,
    Event,
    Future,
    Task,
    create_task,
    gather,
    get_running_loop,
    wait,
)
from base64 import b64decode, urlsafe_b64decode, urlsafe_b64encode
from collections.abc import AsyncGenerator, Callable, Collection, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from itertools import chain
from types import TracebackType
from typing import Any, Final, Self, cast, final
from urllib.parse import ParseResult, urlparse
from uuid import uuid4

from haiway import (
    BasicValue,
    Meta,
    MetaTags,
    Paginated,
    Pagination,
    PaginationToken,
    ValidationError,
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

from draive.models import ModelToolFunctionSpecification, ModelToolParametersSpecification
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
        request_timeout: float | None = None,
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
                async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=request_timeout,
                ) as session:
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
        request_timeout: float | None = None,
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
                async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=request_timeout,
                ) as session:
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
        request_timeout: float | None = None,
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
                    async with ClientSession(
                        read,
                        write,
                        read_timeout_seconds=request_timeout,
                    ) as session:
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
            return Paginated[ResourceReference](
                (),
                pagination=pagination.with_token(None),
            )

        starting_cursor, page_offset = _decode_single_page_cursor(pagination.token)
        remaining: int = pagination.limit
        references: list[ResourceReference] = []
        cursor: str | None = starting_cursor
        offset: int = page_offset
        next_token: str | None = None
        visited_cursors: set[str | None] = {cursor}
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
                if result.next_cursor is None or result.next_cursor in visited_cursors:
                    next_token = None
                    break

                cursor = result.next_cursor
                visited_cursors.add(cursor)
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

            if result.next_cursor is None or result.next_cursor in visited_cursors:
                next_token = None
                break

            cursor = result.next_cursor
            visited_cursors.add(cursor)
            offset = 0
            next_token = cursor

        return Paginated[ResourceReference](
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
        visited_cursors: set[str | None] = {cursor}
        while True:
            request_params: PaginatedRequestParams | None
            if cursor is not None:
                request_params = PaginatedRequestParams(cursor=cursor)

            else:
                request_params = None

            result: ListToolsResult = await self._session.list_tools(params=request_params)
            mcp_tools.extend(result.tools)
            # a cursor seen before means the server cycles, listing can't advance any further
            if result.next_cursor is None or result.next_cursor in visited_cursors:
                break

            cursor = result.next_cursor
            visited_cursors.add(cursor)

        tools: list[Tool] = []
        for mcp_tool in mcp_tools:
            try:
                tools.append(
                    _convert_tool(
                        mcp_tool,
                        tool_call=self._tool_call,
                        source=self.identifier,
                        tags=self.tags,
                    )
                )

            except _UnsupportedSchema as exc:
                # a single tool with an unsupported schema should not hide the others
                ctx.log_warning(
                    f"Skipping MCP tool {mcp_tool.name} of {self.identifier},"
                    f" its input schema is not supported: {exc}"
                )

        return tuple(tools)

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
            *await gather(
                *(
                    _convert_content(part, scope_uri=self._with_uri_identifier)
                    for part in result.content
                )
            )
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
            return _assemble_uri(
                parsed,
                scheme=parsed.scheme,
                netloc=f"{_encode_identifier(self.identifier)}.{parsed.netloc}",
                path=parsed.path,
            )

        # Ensure path starts with /
        path: str = parsed.path
        if not path.startswith("/"):
            path = "/" + path

        return _assemble_uri(
            parsed,
            # default to mcp scheme if empty
            scheme=parsed.scheme or "mcp",
            # use identifier as domain/netloc
            netloc=_encode_identifier(self.identifier),
            path=path,
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

        identifier: str = _encode_identifier(self.identifier)
        if parsed.netloc == identifier:
            # an empty authority is kept as `scheme:///path`, the local `mcp` scheme is dropped
            return _assemble_uri(
                parsed,
                scheme=parsed.scheme if parsed.scheme != "mcp" else "",
                netloc="",
                path=parsed.path,
            )

        if parsed.netloc.startswith(f"{identifier}."):
            return _assemble_uri(
                parsed,
                scheme=parsed.scheme,
                netloc=parsed.netloc[len(identifier) + 1 :],
                path=parsed.path,
            )

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
        try:
            self._session = await prepared

        except BaseException:
            # interrupted or failed before the session was ready, do not leave the owner behind
            self._session_closing.set()
            if not self._session_owner.done():
                self._session_owner.cancel()
                # wait for the teardown without replacing the original failure
                await wait({self._session_owner})

            raise

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


def _encode_identifier(
    identifier: str,
    /,
) -> str:
    # dots separate the identifier from the original authority, escaping them within
    # the identifier keeps the encoded uri unambiguous (`my.server` != `my` + `server`)
    return identifier.replace("%", "%25").replace(".", "%2E")


def _assemble_uri(
    parsed: ParseResult,
    /,
    *,
    scheme: str,
    netloc: str,
    path: str,
) -> str:
    # `urlunparse` restores the `//` authority separator only for schemes it knows, custom
    # schemes like `notes:///docs/a` would lose it - the uri is assembled explicitly instead
    uri: str = f"{scheme}://{netloc}" if scheme else (f"//{netloc}" if netloc else "")
    uri += path
    if parsed.params:
        uri += f";{parsed.params}"

    if parsed.query:
        uri += f"?{parsed.query}"

    if parsed.fragment:
        uri += f"#{parsed.fragment}"

    return uri


async def _convert_content(  # noqa: C901, PLR0911
    content: MCPTextContent
    | MCPImageContent
    | MCPAudioContent
    | MCPResourceLink
    | MCPEmbeddedResource,
    /,
    *,
    scope_uri: Callable[[str], str] | None = None,
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

        case MCPResourceLink() as link:
            return MultimodalContent.of(
                ResourceReference.of(
                    scope_uri(link.uri) if scope_uri is not None else link.uri,
                    name=link.name,
                    description=link.description,
                    mime_type=link.mime_type,
                )
            )


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


class _UnsupportedSchema(Exception):
    """An MCP tool schema which can not be translated without changing its meaning."""


def _tool_parameters(
    name: str,
    input_schema: Mapping[str, Any],
    /,
) -> ModelToolParametersSpecification:
    # server schemas carry keywords the specification does not declare (`default`,
    # `minimum`...), decoding skips them while validating the rest, arguments stay closed
    try:
        parameters: ModelToolParametersSpecification | None = (
            ModelToolFunctionSpecification.from_mapping(
                {
                    "name": name,
                    "parameters": {
                        **_SchemaTranslation(input_schema).translated_arguments(),
                        "additionalProperties": False,
                    },
                },
                ignore_extra=True,
            ).parameters
        )

    except ValidationError as exc:
        raise _UnsupportedSchema(str(exc)) from exc

    if parameters is None:
        raise _UnsupportedSchema("arguments are missing")

    return parameters


_DEFINITIONS: Final[Mapping[str, str]] = {
    "$defs": "#/$defs/",
    "definitions": "#/definitions/",
}
# annotations without an equivalent in the specification, dropped wherever they appear
_DROPPED_KEYWORDS: Final[frozenset[str]] = frozenset(
    ("$comment", "$defs", "$id", "$schema", "definitions", "title")
)
_STRUCTURAL_KEYWORDS: Final[frozenset[str]] = frozenset(
    ("additionalProperties", "enum", "items", "prefixItems", "properties", "required", "type")
)
_ANCHOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][-A-Za-z0-9._]*$")


@final
class _SchemaTranslation:
    """Translate a JSON schema into the form of a type specification, or refuse it.

    References to `$defs` and `definitions` are inlined, a recursive definition is anchored
    where it is inlined first and referenced as `#Name` afterwards, as a recursive State is.
    A reference with sibling keywords and every `allOf` branch are combined into a single
    schema, `oneOf` becomes `anyOf` and `const` a single value `enum`. Objects get an
    explicit `additionalProperties`, closed when they declare properties. Arguments declared
    by root `anyOf` branches become root properties. Anything which can not be translated
    without changing what the schema accepts raises `_UnsupportedSchema`, keywords without
    an equivalent which only constrain values further (`minimum`, `pattern`...) are left
    for decoding to skip.
    """

    __slots__ = (
        "_active",
        "_anchored",
        "_definitions",
        "_recursive",
        "_root",
    )

    def __init__(
        self,
        root: Mapping[str, Any],
    ) -> None:
        self._root: Mapping[str, Any] = root
        self._definitions: dict[str, Mapping[str, Any]] = {}
        for keyword, prefix in _DEFINITIONS.items():
            definitions: Any = root.get(keyword)
            if definitions is None:
                continue

            if not isinstance(definitions, Mapping):
                raise _UnsupportedSchema(f"{keyword} is not a mapping")

            self._definitions[prefix] = cast(Mapping[str, Any], definitions)

        self._active: list[str] = []
        self._recursive: set[str] = set()
        self._anchored: set[str] = set()

    def translated_arguments(self) -> Mapping[str, Any]:
        node: dict[str, Any] = self.translated(self._root, path="$")
        if node.pop("type", "object") != "object":
            raise _UnsupportedSchema("$ is not an object")

        properties: dict[str, list[Any]] = {
            key: [value] for key, value in node.pop("properties", {}).items()
        }
        required: list[str] = list(node.pop("required", ()))
        match node.pop("anyOf", None):
            case None:
                pass

            case branches:
                shared: set[str] | None = None
                for index, branch in enumerate(branches):
                    branch_path: str = f"$.anyOf[{index}]"
                    if branch.get("type", "object") != "object" or "anyOf" in branch:
                        raise _UnsupportedSchema(f"{branch_path} is not a plain object")

                    for key, value in branch.get("properties", {}).items():
                        declared: list[Any] = properties.setdefault(key, [])
                        if value not in declared:
                            declared.append(value)

                    branch_required: set[str] = set(branch.get("required", ()))
                    shared = branch_required if shared is None else shared & branch_required

                required.extend(sorted(shared or ()))

        if not properties and node.get("additionalProperties", False) is not False:
            # arguments of any name can not be declared, closing them would drop all
            if self._root.get("additionalProperties", False) is not False:
                raise _UnsupportedSchema("$ declares open arguments only")

        arguments: dict[str, Any] = {
            **{key: value for key, value in node.items() if key == "description"},
            "type": "object",
            "properties": {
                key: declared[0] if len(declared) == 1 else {"anyOf": declared}
                for key, declared in properties.items()
            },
        }
        if required:
            arguments["required"] = list(dict.fromkeys(required))

        _verify_references(arguments)
        return arguments

    def translated(  # noqa: C901, PLR0912
        self,
        schema: Any,
        /,
        *,
        path: str,
    ) -> dict[str, Any]:
        if not isinstance(schema, Mapping):
            raise _UnsupportedSchema(f"{path} is not a schema")

        node: dict[str, Any] = {
            key: value
            for key, value in cast(Mapping[str, Any], schema).items()
            if key not in _DROPPED_KEYWORDS
        }
        reference: Any = node.pop("$ref", None)
        branches: Any = node.pop("allOf", None)

        if "oneOf" in node:
            if "anyOf" in node:
                raise _UnsupportedSchema(f"{path} declares both anyOf and oneOf")

            node["anyOf"] = node.pop("oneOf")

        if "const" in node:
            constant: Any = node.pop("const")
            enum: Any = node.get("enum")
            if enum is not None and (not _is_sequence(enum) or constant not in enum):
                raise _UnsupportedSchema(f"{path} declares const outside of its enum")

            node["enum"] = [constant]

        if "properties" in node:
            node.setdefault("type", "object")  # properties constrain objects only

        node = self._translated_keywords(node, path=path)

        if reference is not None:
            if not isinstance(reference, str):
                raise _UnsupportedSchema(f"{path}.$ref is not a string")

            node = _combined(self._resolved(reference, path=path), node, path=path)

        if branches is not None:
            if not _is_sequence(branches) or not branches:
                raise _UnsupportedSchema(f"{path}.allOf is not a sequence of schemas")

            for index, branch in enumerate(cast(Sequence[Any], branches)):
                node = _combined(
                    self.translated(branch, path=f"{path}.allOf[{index}]"),
                    node,
                    path=path,
                )

        if "$ref" in node and node.keys() - {"$ref", "description"}:
            raise _UnsupportedSchema(f"{path} constrains a recursive reference")

        # alternatives alongside other structure require both, which a specification can not
        # express - the root is the exception as its alternatives become arguments
        if path != "$" and "anyOf" in node and node.keys() & _STRUCTURAL_KEYWORDS:
            raise _UnsupportedSchema(f"{path} constrains its alternatives")

        if node.get("type") == "object":
            if "properties" in node:
                # declared properties close the object, as the arguments are closed
                node["additionalProperties"] = False

            else:
                # an object without properties is open unless it declares its values
                node.setdefault("additionalProperties", True)

        return node

    def _translated_keywords(  # noqa: C901
        self,
        node: Mapping[str, Any],
        /,
        *,
        path: str,
    ) -> dict[str, Any]:
        translated: dict[str, Any] = {}
        for keyword, value in node.items():
            match keyword:
                case "properties":
                    if not isinstance(value, Mapping):
                        raise _UnsupportedSchema(f"{path}.properties is not a mapping")

                    translated[keyword] = {
                        key: self.translated(element, path=f"{path}.{key}")
                        for key, element in cast(Mapping[str, Any], value).items()
                    }

                case "required":
                    if not _is_sequence(value) or not all(
                        isinstance(element, str) for element in cast(Sequence[Any], value)
                    ):
                        raise _UnsupportedSchema(f"{path}.required is not a sequence of names")

                    translated[keyword] = list(cast(Sequence[str], value))

                case "items" | "additionalProperties" if isinstance(value, bool):
                    translated[keyword] = value

                case "items" | "additionalProperties":
                    translated[keyword] = self.translated(value, path=f"{path}.{keyword}")

                case "anyOf" | "prefixItems":
                    if not _is_sequence(value) or not value:
                        raise _UnsupportedSchema(f"{path}.{keyword} is not a sequence of schemas")

                    translated[keyword] = [
                        self.translated(element, path=f"{path}.{keyword}[{index}]")
                        for index, element in enumerate(cast(Sequence[Any], value))
                    ]

                case "enum":
                    if not _is_sequence(value):
                        raise _UnsupportedSchema(f"{path}.enum is not a sequence")

                    translated[keyword] = list(cast(Sequence[Any], value))

                case _:
                    translated[keyword] = value

        return translated

    def _resolved(
        self,
        reference: str,
        /,
        *,
        path: str,
    ) -> dict[str, Any]:
        for prefix, definitions in self._definitions.items():
            if not reference.startswith(prefix):
                continue

            name: str = reference.removeprefix(prefix)
            if "/" in name or name not in definitions:
                break

            # names of both dialects can repeat, anchors have to stay distinct
            anchor: str = name if prefix == _DEFINITIONS["$defs"] else f"definitions.{name}"
            if anchor in self._active:
                if not _ANCHOR_PATTERN.match(anchor):
                    raise _UnsupportedSchema(f"{path} recursive definition can not be anchored")

                self._recursive.add(anchor)
                return {"$ref": f"#{anchor}"}

            if anchor in self._anchored:
                return {"$ref": f"#{anchor}"}

            self._active.append(anchor)
            try:
                resolved: dict[str, Any] = self.translated(
                    definitions[name],
                    path=f"{path}.$ref",
                )

            finally:
                self._active.pop()

            if anchor in self._recursive:
                self._anchored.add(anchor)
                return {**resolved, "$anchor": anchor}

            return resolved

        raise _UnsupportedSchema(f"{path} references {reference!r} which is not a definition")


def _combined(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    /,
    *,
    path: str,
) -> dict[str, Any]:
    # both schemas apply - properties and requirements of both are kept, a keyword declared
    # differently by them can not be expressed, descriptions of the second take precedence
    combined: dict[str, Any] = dict(first)
    for keyword, value in second.items():
        current: Any = combined.get(keyword)
        if keyword not in combined or current == value or keyword == "description":
            combined[keyword] = value

        elif keyword == "properties":
            properties: dict[str, Any] = dict(current)
            for key, element in value.items():
                if key in properties and properties[key] != element:
                    raise _UnsupportedSchema(f"{path}.{key} is declared differently")

                properties[key] = element

            combined[keyword] = properties

        elif keyword == "required":
            combined[keyword] = list(dict.fromkeys((*current, *value)))

        elif keyword == "additionalProperties" and False in (current, value):
            combined[keyword] = False

        else:
            raise _UnsupportedSchema(f"{path}.{keyword} is declared differently")

    return combined


def _verify_references(
    arguments: Mapping[str, Any],
    /,
) -> None:
    anchors: set[str] = set()
    references: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, element in cast(Mapping[str, Any], value).items():
                if key == "$anchor":
                    anchors.add(element)

                elif key == "$ref":
                    references.add(element)

                else:
                    collect(element)

        elif _is_sequence(value):
            for element in cast(Sequence[Any], value):
                collect(element)

    collect(arguments)
    if dangling := {reference for reference in references if reference[1:] not in anchors}:
        raise _UnsupportedSchema(f"references {sorted(dangling)} lost their definitions")


def _is_sequence(
    value: Any,
    /,
) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


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
        parameters=_tool_parameters(name, mcp_tool.input_schema),
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
        all_clients: Sequence[MCPClient] = [client, *clients]
        identifiers: set[str] = set()
        for candidate in all_clients:
            if candidate.identifier in identifiers:
                raise ValueError(f"Duplicate MCP client identifier: {candidate.identifier}")

            identifiers.add(candidate.identifier)

        self._clients: Mapping[str, MCPClient] = {c.identifier: c for c in all_clients}
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
            return Paginated[ResourceReference](
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

        return Paginated[ResourceReference](
            tuple(references[: pagination.limit]),
            pagination=pagination.with_token(next_token),
        )

    async def resource_fetch(
        self,
        uri: str,
        **extra: Any,
    ) -> Collection[ResourceReference] | ResourceContent | None:
        resources: ResourcesRepository | None = (
            self._resources.get(client_identifier)
            if (client_identifier := self._client_identifier_for_uri(uri)) is not None
            else None
        )
        if resources is not None:
            return await resources.fetching(uri)

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

        for identifier in self._clients:
            if _encode_identifier(identifier) == parsed.netloc:
                return identifier

        # a server may scope its resources under a dotted prefix, i.e. `mcp://server.item`
        encoded_identifier: str = parsed.netloc.split(".", 1)[0]
        for identifier in self._clients:
            if _encode_identifier(identifier) == encoded_identifier:
                return identifier

        return None

    async def tools_fetch(
        self,
        *,
        mcp_server: str | None = None,
        **extra: Any,
    ) -> Sequence[Tool]:
        if mcp_server is None:
            aggregated: tuple[Tool, ...] = tuple(
                chain.from_iterable(
                    await gather(*[client.load(**extra) for client in self._tools.values()])
                )
            )
            seen_names: set[str] = set()
            for tool in aggregated:
                if tool.name in seen_names:
                    # toolboxes keep the last duplicate, select a server to avoid ambiguity
                    ctx.log_warning(
                        f"MCP tool '{tool.name}' is provided by multiple servers,"
                        " only the last one will be available"
                    )

                seen_names.add(tool.name)

            return aggregated

        elif tools := self._tools.get(mcp_server):
            return await tools.load(**extra)

        else:
            return ()

    async def _enter_clients(self) -> Sequence[Sequence[ResourcesRepository | ToolsProvider]]:
        clients: Sequence[MCPClient] = tuple(self._clients.values())
        entering: Sequence[Task[Sequence[ResourcesRepository | ToolsProvider]]] = tuple(
            create_task(client.__aenter__()) for client in clients
        )
        try:
            await wait(entering, return_when=FIRST_EXCEPTION)
            # `wait` does not raise, a failed entry has to be surfaced explicitly
            for task in entering:
                if task.done() and not task.cancelled() and (failure := task.exception()):
                    raise failure

        except BaseException:
            # failure or cancellation - stop pending entries and release the ones which entered,
            # `__aexit__` is not called after a failed enter
            for task in entering:
                task.cancel()

            await wait(entering)
            await gather(
                *[
                    client.__aexit__(None, None, None)
                    for client, task in zip(clients, entering, strict=True)
                    if not task.cancelled() and task.exception() is None
                ],
                return_exceptions=True,
            )
            raise

        return [task.result() for task in entering]

    async def __aenter__(self) -> Sequence[ResourcesRepository | ToolsProvider]:
        features: Sequence[
            Sequence[ResourcesRepository | ToolsProvider]
        ] = await self._enter_clients()

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
        # every client has to be released, even if some of them fail to close
        results: Sequence[BaseException | None] = await gather(
            *[
                client.__aexit__(
                    exc_type,
                    exc_val,
                    exc_tb,
                )
                for client in self._clients.values()
            ],
            return_exceptions=True,
        )
        if failures := [result for result in results if isinstance(result, BaseException)]:
            if len(failures) == 1:
                raise failures[0]

            raise BaseExceptionGroup("Failed to close MCP clients", failures)
