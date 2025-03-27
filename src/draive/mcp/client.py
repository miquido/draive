from asyncio import gather
from base64 import b64decode
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from itertools import chain
from types import TracebackType
from typing import Any, Self, final
from urllib.parse import ParseResult, urlparse, urlunparse
from uuid import uuid4

from haiway import as_dict, as_list, ctx
from haiway.utils.freezing import freeze
from mcp import ClientSession, GetPromptResult, ListToolsResult, StdioServerParameters, stdio_client
from mcp import Tool as MCPTool
from mcp.client.sse import sse_client
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ListResourcesResult,
    ReadResourceResult,
    TextResourceContents,
)
from mcp.types import ImageContent as MCPImageContent
from mcp.types import TextContent as MCPTextContent
from pydantic import AnyUrl

from draive.lmm import LMMCompletion, LMMContextElement, LMMInput
from draive.lmm.types import LMMToolError
from draive.multimodal import MediaContent, MultimodalContent, TextContent
from draive.parameters import BasicValue
from draive.parameters.model import DataModel
from draive.parameters.specification import validated_specification
from draive.prompts import Prompt, PromptDeclaration, PromptDeclarationArgument, PromptRepository
from draive.resources import Resource, ResourceContent, ResourceDeclaration, Resources
from draive.tools import AnyTool, ExternalTools, Tool

__all__ = [
    "MCPClient",
    "MCPClientAggregate",
]


@final
class MCPClient:
    @classmethod
    def stdio(
        cls,
        identifier: str | None = None,
        *,
        command: str,
        args: Sequence[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> Self:
        @asynccontextmanager
        async def mcp_stdio_session() -> AsyncGenerator[ClientSession]:
            async with stdio_client(
                StdioServerParameters(
                    command=command,
                    args=as_list(args) if args else [],
                    env=env,
                )
            ) as (read, write):
                # verify typing here
                async with ClientSession(read, write) as session:  # pyright: ignore[reportArgumentType]
                    yield session

        return cls(identifier, session_manager=mcp_stdio_session())

    @classmethod
    def sse(
        cls,
        identifier: str | None = None,
        *,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = 5,
        sse_read_timeout: float = 60 * 5,
    ) -> Self:
        @asynccontextmanager
        async def mcp_sse_session() -> AsyncGenerator[ClientSession]:
            async with sse_client(
                url=url,
                headers=headers,
                timeout=timeout,
                sse_read_timeout=sse_read_timeout,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session

        return cls(identifier, session_manager=mcp_sse_session())

    def __init__(
        self,
        identifier: str | None,
        *,
        session_manager: AbstractAsyncContextManager[ClientSession],
    ) -> None:
        self.identifier: str = identifier or uuid4().hex
        self._session_manager: AbstractAsyncContextManager[ClientSession] = session_manager
        self._session: ClientSession

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
                meta={"source": self.identifier},
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
                    name="resource",  # TODO: resource name?
                    description=None,
                    content=resources,
                    meta={"source": self.identifier},
                )

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
                meta={"source": self.identifier},
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

        return Prompt(
            name=name,
            description=fetched_prompt.description,
            content=prompt_context,
            meta={"source": self.identifier},
        )

    async def tools_fetch(
        self,
        **extra: Any,
    ) -> Sequence[AnyTool]:
        tools: ListToolsResult = await self._session.list_tools()

        return tuple(
            _convert_tool(
                tool,
                tool_call=self._tool_call,
                source=self.identifier,
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
                    name="resource",  # TODO: resource name?
                    description=None,
                    content=ResourceContent(
                        blob=text_resource.text.encode(),
                        mime_type=text_resource.mimeType or "text/plain",
                    ),
                    meta={"source": self.identifier},
                )

            case BlobResourceContents() as blob_resource:
                return Resource(
                    uri=self._with_uri_identifier(resource.uri.unicode_string()),
                    name="resource",  # TODO: resource name?
                    description=None,
                    content=ResourceContent(
                        # TODO: to verify, it seems strange but did not worked otherwise when tested
                        blob=b64decode(b64decode(blob_resource.blob)),
                        mime_type=blob_resource.mimeType or "application/octet-stream",
                    ),
                    meta={"source": self.identifier},
                )

    async def __aenter__(self) -> tuple[Resources, PromptRepository, ExternalTools]:
        self._session = await self._session_manager.__aenter__()
        await self._session.initialize()

        return (
            Resources(
                list_fetching=self.resources_list,
                fetching=self.resource_fetch,
            ),
            PromptRepository(
                list=self.prompts_list,
                fetch=self.prompt_fetch,
            ),
            ExternalTools(
                fetch=self.tools_fetch,
            ),
        )

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


async def _convert_content(
    content: MCPTextContent | MCPImageContent | EmbeddedResource,
    /,
) -> MultimodalContent:
    match content:
        case MCPTextContent() as text:
            return MultimodalContent.of(text.text)

        case MCPImageContent() as image:
            return MultimodalContent.of(
                MediaContent.base64(
                    image.data,
                    media=image.mimeType,
                )
            )

        case EmbeddedResource() as resource:
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
                                TextContent(text=b64decode(blob.blob).decode())
                            )

                        case "application/json":
                            return MultimodalContent.of(
                                DataModel.from_json(b64decode(blob.blob).decode())
                            )

                        case other:
                            # try to match supported media or raise an exception
                            return MultimodalContent.of(
                                MediaContent.data(
                                    b64decode(blob.blob),
                                    media=other,
                                )
                            )


def _convert_tool(
    mcp_tool: MCPTool,
    /,
    *,
    tool_call: Callable[[str, Mapping[str, BasicValue]], Coroutine[None, None, MultimodalContent]],
    source: str,
) -> AnyTool:
    name: str = mcp_tool.name

    async def remote_call(**arguments: Any) -> MultimodalContent:
        return await tool_call(
            name,
            arguments,
        )

    return Tool(
        name=name,
        description=mcp_tool.description,
        specification=validated_specification(mcp_tool.inputSchema),
        function=remote_call,
        availability_check=None,
        format_result=_format_tool_result,
        format_failure=_format_tool_failure,
        direct_result=False,
        meta={"source": source},
    )


def _format_tool_result(
    result: Any,
) -> MultimodalContent:
    assert isinstance(result, MultimodalContent)  # nosec: B101
    return result


def _format_tool_failure(
    exception: Exception,
) -> MultimodalContent:
    if isinstance(exception, LMMToolError):
        return exception.content

    return MultimodalContent.of("ERROR")


@final
class MCPClientAggregate:
    def __init__(
        self,
        client: MCPClient,
        *clients: MCPClient,
    ) -> None:
        self._clients: Mapping[str, MCPClient] = {c.identifier: c for c in [client, *clients]}

        freeze(self)

    async def resources_list(
        self,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]:
        return tuple(
            chain.from_iterable(
                await gather(*[client.resources_list(**extra) for client in self._clients.values()])
            )
        )

    async def resource_fetch(
        self,
        uri: str,
        **extra: Any,
    ) -> Resource | None:
        if client := self._client_for_uri(uri):
            return await client.resource_fetch(uri)

        else:
            ctx.log_warning(f"Requested resource ({uri}) from unknown source")
            return None

    def _client_for_uri(
        self,
        uri: str,
        /,
    ) -> MCPClient | None:
        """Find server associated with URI."""
        if not uri:
            return None

        parsed: ParseResult = urlparse(uri)
        if not parsed.netloc:
            return None

        if parsed.netloc in self._clients.keys():
            return self._clients.get(parsed.netloc)

        match parsed.netloc.split(".", 1):
            case [identifier, _]:
                return self._clients.get(identifier)

            case _:
                return None

    async def prompts_list(
        self,
        **extra: Any,
    ) -> Sequence[PromptDeclaration]:
        return tuple(
            chain.from_iterable(
                await gather(*[client.prompts_list(**extra) for client in self._clients.values()])
            )
        )

    async def prompt_fetch(
        self,
        name: str,
        *,
        arguments: Mapping[str, str] | None,
        **extra: Any,
    ) -> Prompt | None:
        if client := self._client_for_prompt_name(name):
            return await client.prompt_fetch(
                name,
                arguments=arguments,
            )

        else:
            ctx.log_warning(f"Requested prompt ({name}) from unknown source")
            return None

    def _client_for_prompt_name(
        self,
        name: str,
        /,
    ) -> MCPClient | None:
        """Find server associated with prompt name."""
        if not name:
            return None

        match name.split("|", 1):
            case [identifier, _]:
                return self._clients.get(identifier)

            case _:
                return None

    async def tools_fetch(
        self,
        **extra: Any,
    ) -> Sequence[AnyTool]:
        return tuple(
            chain.from_iterable(
                await gather(*[client.tools_fetch(**extra) for client in self._clients.values()])
            )
        )

    async def __aenter__(self) -> tuple[Resources, PromptRepository, ExternalTools]:
        await gather(*[client.__aenter__() for client in self._clients.values()])

        return (
            Resources(
                list_fetching=self.resources_list,
                fetching=self.resource_fetch,
            ),
            PromptRepository(
                list=self.prompts_list,
                fetch=self.prompt_fetch,
            ),
            ExternalTools(
                fetch=self.tools_fetch,
            ),
        )

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
