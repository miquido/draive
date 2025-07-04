from asyncio import gather
from base64 import urlsafe_b64decode
from collections.abc import AsyncGenerator, Callable, Collection, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from itertools import chain
from types import TracebackType
from typing import Any, Literal, Self, cast, final
from urllib.parse import ParseResult, urlparse, urlunparse
from uuid import uuid4

from haiway import as_dict, as_list, ctx
from mcp import ClientSession, GetPromptResult, ListToolsResult, StdioServerParameters, stdio_client
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

from draive.commons import Meta, MetaTags
from draive.lmm import LMMCompletion, LMMContextElement, LMMInput, LMMToolError
from draive.multimodal import MediaData, MultimodalContent, TextContent
from draive.parameters import BasicValue, DataModel, validated_specification
from draive.prompts import Prompt, PromptDeclaration, PromptDeclarationArgument, Prompts
from draive.resources import Resource, ResourceContent, ResourceDeclaration, Resources
from draive.tools import FunctionTool, Tool, Tools

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
        features: Collection[Literal["resources", "prompts", "tools"]] | None = None,
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
            features=features if features is not None else {"resources", "prompts", "tools"},
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
        features: Collection[Literal["resources", "prompts", "tools"]] | None = None,
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
            features=features if features is not None else {"resources", "prompts", "tools"},
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
        features: Collection[Literal["resources", "prompts", "tools"]],
        tags: MetaTags,
    ) -> None:
        self.identifier: str = identifier
        self._session_manager: AbstractAsyncContextManager[ClientSession] = session_manager
        self._session: ClientSession
        self._features: Collection[Literal["resources", "prompts", "tools"]] = features
        self.tags: MetaTags = tags

    async def resources_list(
        self,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized through async context entering"

        # TODO: in theory there are some extra elements like pagination
        # we are not supporting it as for now (how to pass cursor anyways?)
        result: ListResourcesResult = await self._session.list_resources()

        return [
            ResourceDeclaration(
                uri=self._with_uri_identifier(resource.uri.unicode_string()),
                name=resource.name,
                description=resource.description,
                mime_type=resource.mimeType,
                meta=Meta(
                    {
                        "mcp_server": self.identifier,
                        "tags": self.tags,
                    }
                ),
            )
            for resource in result.resources
        ]

    async def resource_fetch(
        self,
        uri: str,
        **extra: Any,
    ) -> Resource | None:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized through async context entering"
        result: ReadResourceResult = await self._session.read_resource(
            uri=AnyUrl(self._without_uri_identifier(uri))
        )

        match [self._convert_resource_content(element) for element in result.contents]:
            case [resource]:
                # if there is only a single element return it directly
                return resource

            case [*resources]:
                return Resource(
                    uri=uri,
                    name="resources",
                    description=None,
                    content=resources,
                    meta=Meta(
                        {
                            "mcp_server": self.identifier,
                            "tags": self.tags,
                        }
                    ),
                )

    async def resource_upload(
        self,
        resource: Resource,
        **extra: Any,
    ) -> None:
        raise NotImplementedError("Resource uploading is not supported by MCP servers")

    async def prompts_list(
        self,
        **extra: Any,
    ) -> Sequence[PromptDeclaration]:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized through async context entering"

        # TODO: in theory there are some extra elements like pagination
        # we are not supporting it as for now (how to pass cursor anyways?)
        return tuple(
            PromptDeclaration(
                name=self._with_prompt_name_identifier(prompt.name),
                description=prompt.description,
                arguments=tuple(
                    PromptDeclarationArgument(
                        name=argument.name,
                        specification={
                            "type": "string",
                        }
                        if argument.description is None
                        else {
                            "type": "string",
                            "description": argument.description,
                        },
                        required=argument.required if argument.required is not None else True,
                    )
                    for argument in prompt.arguments
                )
                if prompt.arguments
                else (),
                meta=Meta(
                    {
                        "mcp_server": self.identifier,
                        "tags": self.tags,
                    }
                ),
            )
            for prompt in (await self._session.list_prompts()).prompts
        )

    async def prompt_fetch(
        self,
        name: str,
        *,
        arguments: Mapping[str, str] | None,
        **extra: Any,
    ) -> Prompt:
        assert hasattr(  # nosec: B101
            self, "_session"
        ), "MCPClient has to be initialized through async context entering"

        fetched_prompt: GetPromptResult = await self._session.get_prompt(
            name=self._without_prompt_name_identifier(name),
            arguments=as_dict(arguments) if arguments is not None else None,
        )

        prompt_context: list[LMMContextElement] = []
        for element in fetched_prompt.messages:
            match element.role:
                case "user":
                    prompt_context.append(LMMInput.of(await _convert_content(element.content)))

                case "assistant":
                    prompt_context.append(LMMCompletion.of(await _convert_content(element.content)))

                case _:
                    raise ValueError(f"Unsupported prompt element role: {element.role}")

        return Prompt(
            name=name,
            description=fetched_prompt.description,
            content=prompt_context,
            meta=Meta(
                {
                    "mcp_server": self.identifier,
                    "tags": self.tags,
                }
            ),
        )

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
            raise LMMToolError(
                f"Remote tool {name} failed",
                content=content,
            )

        return content

    def _with_prompt_name_identifier(
        self,
        name: str,
        /,
    ) -> str:
        """Add server identifier to prompt name."""

        return f"{self.identifier}|{name}"

    def _without_prompt_name_identifier(
        self,
        name: str,
        /,
    ) -> str:
        """Remove server identifier from prompt name."""

        match name.split("|", 1):
            case [identifier, reminder] if identifier == self.identifier:
                return reminder

            case _:
                return name

    def _with_uri_identifier(
        self,
        uri: str,
        /,
    ) -> str:
        """Add server identifier to URI."""
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
                    parsed.scheme or "mcpclient",
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
        """Remove server identifier from URI."""
        if not uri:
            return uri

        parsed: ParseResult = urlparse(uri)
        if not parsed.netloc:
            return uri

        if parsed.netloc == self.identifier:
            return urlunparse(
                (
                    parsed.scheme if parsed.scheme != "mcpclient" else "",
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
    ) -> Resource:
        match resource:
            case TextResourceContents() as text_resource:
                return Resource(
                    uri=self._with_uri_identifier(resource.uri.unicode_string()),
                    name="resource",
                    description=None,
                    content=ResourceContent(
                        blob=text_resource.text.encode(),
                        mime_type=text_resource.mimeType or "text/plain",
                    ),
                    meta=Meta(
                        {
                            "mcp_server": self.identifier,
                            "tags": self.tags,
                        }
                    ),
                )

            case BlobResourceContents() as blob_resource:
                return Resource(
                    uri=self._with_uri_identifier(resource.uri.unicode_string()),
                    name="resource",
                    description=None,
                    content=ResourceContent(
                        blob=urlsafe_b64decode(blob_resource.blob),
                        mime_type=blob_resource.mimeType or "application/octet-stream",
                    ),
                    meta=Meta(
                        {
                            "mcp_server": self.identifier,
                            "tags": self.tags,
                        }
                    ),
                )

    async def __aenter__(self) -> Sequence[Resources | Prompts | Tools]:
        self._session = await self._session_manager.__aenter__()
        await self._session.initialize()

        features: list[Resources | Prompts | Tools] = []
        if "resources" in self._features:
            features.append(
                Resources(
                    list_fetching=self.resources_list,
                    fetching=self.resource_fetch,
                    uploading=self.resource_upload,
                    meta=Meta(
                        {
                            "mcp_server": self.identifier,
                            "tags": self.tags,
                        }
                    ),
                )
            )

        if "prompts" in self._features:
            features.append(
                Prompts(
                    list_fetching=self.prompts_list,
                    fetching=self.prompt_fetch,
                    meta=Meta(
                        {
                            "mcp_server": self.identifier,
                            "tags": self.tags,
                        }
                    ),
                )
            )
        if "tools" in self._features:
            features.append(
                Tools(
                    fetching=self.tools_fetch,
                    meta=Meta(
                        {
                            "mcp_server": self.identifier,
                            "tags": self.tags,
                        }
                    ),
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
                MediaData.of(
                    urlsafe_b64decode(image.data),
                    media=image.mimeType,
                )
            )

        case MCPAudioContent() as audio:
            return MultimodalContent.of(
                MediaData.of(
                    urlsafe_b64decode(audio.data),
                    media=audio.mimeType,
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
                                "Unsuppoprted MCP prompt message element - missing mime!"
                            )

                        case "text/plain":
                            return MultimodalContent.of(
                                TextContent(text=urlsafe_b64decode(blob.blob).decode())
                            )

                        case "application/json":
                            return MultimodalContent.of(
                                DataModel.from_json(urlsafe_b64decode(blob.blob).decode())
                            )

                        case other:
                            # try to match supported media or raise an exception
                            return MultimodalContent.of(
                                MediaData.of(
                                    urlsafe_b64decode(blob.blob),
                                    media=other,
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
        parameters=validated_specification(mcp_tool.inputSchema),
        function=remote_call,
        availability_check=None,
        format_result=_format_tool_result,
        format_failure=_format_tool_failure,
        handling="auto",
        meta=Meta(
            {
                "mcp_server": source,
                "tags": tags,
            }
        ),
    )


def _format_tool_result(
    result: Any,
) -> MultimodalContent:
    assert isinstance(result, MultimodalContent)  # nosec: B101
    return result


def _format_tool_failure(
    error: Exception,
) -> MultimodalContent:
    if isinstance(error, LMMToolError):
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
        self._resources: Mapping[str, Resources]
        self._prompts: Mapping[str, Prompts]
        self._tools: Mapping[str, Tools]

    async def resources_list(
        self,
        *,
        mcp_server: str | None = None,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]:
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
    ) -> Resource | None:
        if client_identifier := self._client_identifier_for_uri(uri):
            return await self._resources[client_identifier].fetch(uri)

        else:
            ctx.log_warning(f"Requested resource ({uri}) from unknown source")
            return None

    async def resource_upload(
        self,
        resource: Resource,
        **extra: Any,
    ) -> None:
        raise NotImplementedError("Resource uploading is not supported by MCP servers")

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

    async def prompts_list(
        self,
        *,
        mcp_server: str | None = None,
        **extra: Any,
    ) -> Sequence[PromptDeclaration]:
        if mcp_server is None:
            return tuple(
                chain.from_iterable(
                    await gather(*[client.fetch_list(**extra) for client in self._prompts.values()])
                )
            )

        elif prompts := self._prompts.get(mcp_server):
            return await prompts.fetch_list(**extra)

        else:
            return ()

    async def prompt_fetch(
        self,
        name: str,
        *,
        arguments: Mapping[str, str] | None,
        **extra: Any,
    ) -> Prompt | None:
        if client_identifier := self._client_identifier_for_prompt_name(name):
            return await self._prompts[client_identifier].fetch(
                name,
                arguments=arguments,
            )

        else:
            ctx.log_warning(f"Requested prompt ({name}) from unknown source")
            return None

    def _client_identifier_for_prompt_name(
        self,
        name: str,
        /,
    ) -> str | None:
        if not name:
            return None

        match name.split("|", 1):
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
                    await gather(*[client.fetch(**extra) for client in self._tools.values()])
                )
            )

        elif tools := self._tools.get(mcp_server):
            return await tools.fetch(**extra)

        else:
            return ()

    async def __aenter__(self) -> Sequence[Resources | Prompts | Tools]:
        features: Sequence[Sequence[Resources | Prompts | Tools]] = await gather(
            *[client.__aenter__() for client in self._clients.values()]
        )

        self._resources = {}
        self._prompts = {}
        self._tools = {}

        for states in features:
            for state in states:
                match state:
                    case Resources() as resources:
                        self._resources[cast(str, resources.meta["mcp_server"])] = resources

                    case Prompts() as prompts:
                        self._prompts[cast(str, prompts.meta["mcp_server"])] = prompts

                    case Tools() as tools:
                        self._tools[cast(str, tools.meta["mcp_server"])] = tools

        inherited_features: list[Resources | Prompts | Tools] = []
        if self._resources:
            inherited_features.append(
                Resources(
                    list_fetching=self.resources_list,
                    fetching=self.resource_fetch,
                    uploading=self.resource_upload,
                    meta=Meta({"mcp_server": "mcp_aggregate"}),
                )
            )

        if self._prompts:
            inherited_features.append(
                Prompts(
                    list_fetching=self.prompts_list,
                    fetching=self.prompt_fetch,
                    meta=Meta({"mcp_server": "mcp_aggregate"}),
                )
            )

        if self._tools:
            inherited_features.append(
                Tools(
                    fetching=self.tools_fetch,
                    meta=Meta({"mcp_server": "mcp_aggregate"}),
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
