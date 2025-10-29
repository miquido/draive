from asyncio import gather
from base64 import urlsafe_b64decode
from collections.abc import AsyncGenerator, Callable, Collection, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from itertools import chain
from types import TracebackType
from typing import Any, Self, cast, final
from urllib.parse import ParseResult, urlparse, urlunparse
from uuid import uuid4

from haiway import BasicValue, Meta, MetaTags, as_dict, as_list, as_tuple, ctx
from mcp import ClientSession, ListToolsResult, StdioServerParameters, stdio_client
from mcp import Tool as MCPTool
from mcp.client.sse import sse_client
from mcp.types import AudioContent as MCPAudioContent
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ListResourcesResult,
    ReadResourceResult,
    TextResourceContents,
)
from mcp.types import EmbeddedResource as MCPEmbeddedResource
from mcp.types import ImageContent as MCPImageContent
from mcp.types import ResourceLink as MCPResourceLink
from mcp.types import TextContent as MCPTextContent
from pydantic import AnyUrl

from draive.models import (
    FunctionTool,
    ModelToolParametersSpecification,
    ModelToolSpecification,
    Tool,
    ToolError,
    ToolsProvider,
)
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.parameters import DataModel
from draive.resources import ResourceContent, ResourceReference, ResourcesRepository

__all__ = (
    "MCPClient",
    "MCPClients",
)


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
            identifier or uuid4().hex,
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
            identifier or uuid4().hex,
            session_manager=mcp_sse_session(),
            features=features if features is not None else (ResourcesRepository, ToolsProvider),
            tags=tags if tags is not None else (),
        )

    __slots__ = (
        "_features",
        "_session",
        "_session_manager",
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
        **extra: Any,
    ) -> Sequence[ResourceReference]:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized through async context entering"

        # TODO: in theory there are some extra elements like pagination
        # we are not supporting it as for now (how to pass cursor anyways?)
        result: ListResourcesResult = await self._session.list_resources()

        return [
            ResourceReference(
                uri=self._with_uri_identifier(resource.uri.unicode_string()),
                mime_type=resource.mimeType,
                meta=self._meta(
                    name=resource.name,
                    description=resource.description,
                ),
            )
            for resource in result.resources
        ]

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
            uri=AnyUrl(self._without_uri_identifier(uri))
        )

        match result.contents:
            case [resource]:
                # if there is only a single element return it directly
                return self._convert_resource_content(resource)

            case [*resources]:
                # otherwise convert to references ignoring content
                return [
                    ResourceReference(
                        uri=resource.uri.unicode_string(),
                        mime_type=resource.mimeType,
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
        tools: ListToolsResult = await self._session.list_tools()

        return tuple(
            _convert_tool(
                tool,
                tool_call=self._tool_call,
                source=self.identifier,
                tags=self.tags,
            )
            for tool in tools.tools
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
        content: MultimodalContent = MultimodalContent.of(
            *await gather(*(_convert_content(part) for part in result.content))
        )

        if result.isError:
            raise ToolError(
                f"Remote tool {name} failed",
                content=content,
            )

        return content

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
                return ResourceContent(
                    data=urlsafe_b64decode(text_resource.text.encode()).decode(),
                    mime_type=text_resource.mimeType or "text/plain",
                    meta=self._meta(),
                )

            case BlobResourceContents() as blob_resource:
                return ResourceContent(
                    data=blob_resource.blob,
                    mime_type=blob_resource.mimeType or "application/octet-stream",
                    meta=self._meta(),
                )

    async def __aenter__(self) -> Sequence[ResourcesRepository | ToolsProvider]:
        self._session = await self._session_manager.__aenter__()
        await self._session.initialize()

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

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._session_manager.__aexit__(
            exc_type,
            exc_val,
            exc_tb,
        )
        del self._session


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
                    urlsafe_b64decode(image.data),
                    mime_type=image.mimeType,
                )
            )

        case MCPAudioContent() as audio:
            return MultimodalContent.of(
                ResourceContent.of(
                    urlsafe_b64decode(audio.data),
                    mime_type=audio.mimeType,
                )
            )

        case MCPEmbeddedResource() as resource:
            match resource.resource:
                case TextResourceContents() as text:
                    return MultimodalContent.of(TextContent(text=text.text))

                case BlobResourceContents() as blob:
                    match blob.mimeType:
                        case None:
                            raise NotImplementedError(
                                "Unsupported embedded resource - missing mime!"
                            )

                        case "text/plain":
                            return MultimodalContent.of(
                                TextContent(text=urlsafe_b64decode(blob.blob).decode())
                            )

                        case "application/json":
                            return MultimodalContent.of(
                                ArtifactContent.of(
                                    DataModel.from_json(urlsafe_b64decode(blob.blob).decode())
                                )
                            )

                        case other:
                            # try to match supported media or raise an exception
                            return MultimodalContent.of(
                                ResourceContent.of(
                                    urlsafe_b64decode(blob.blob),
                                    mime_type=other,
                                )
                            )

        case MCPResourceLink():
            raise NotImplementedError("MCP resource links are not supported yet")


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

    return FunctionTool(
        name=name,
        description=mcp_tool.description,
        parameters=cast(
            ModelToolParametersSpecification,
            {
                **mcp_tool.inputSchema,
                "additionalProperties": False,
            },
        ),
        function=remote_call,
        availability=_available,
        result_formatting=_format_tool_result,
        error_formatting=_format_tool_failure,
        handling="response",
        meta=Meta.of(
            {
                "mcp_server": source,
                "tags": as_tuple(tags),
            }
        ),
    )


def _available(
    tools_turn: int,
    specification: ModelToolSpecification,
) -> bool:
    return True


def _format_tool_result(
    result: Any,
) -> MultimodalContent:
    assert isinstance(result, MultimodalContent)  # nosec: B101
    return result


def _format_tool_failure(
    error: Exception,
) -> MultimodalContent:
    if isinstance(error, ToolError):
        return error.content

    return MultimodalContent.of("ERROR")


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
        **extra: Any,
    ) -> Sequence[ResourceReference]:
        if mcp_server is None:
            return tuple(
                chain.from_iterable(
                    await gather(
                        *[client.fetch_list(**extra) for client in self._resources.values()]
                    )
                )
            )

        elif resources := self._resources.get(mcp_server):
            return await resources.fetch_list(**extra)

        else:
            return ()

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
                    await gather(*[client.loading(**extra) for client in self._tools.values()])
                )
            )

        elif tools := self._tools.get(mcp_server):
            return await tools.loading(**extra)

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
