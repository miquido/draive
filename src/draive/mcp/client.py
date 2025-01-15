from asyncio import gather
from base64 import b64decode
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import TracebackType
from typing import Any, Self, cast, final

from haiway import as_dict, as_list
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
from draive.multimodal import MediaContent, MultimodalContent, TextContent, validated_media_type
from draive.parameters import BasicValue, ParametersSpecification
from draive.parameters.model import DataModel
from draive.prompts import Prompt, PromptDeclaration, PromptDeclarationArgument, PromptRepository
from draive.resources import Resource, ResourceContent, ResourceDeclaration, ResourceRepository
from draive.tools import AnyTool, ExternalToolbox, Tool, Toolbox

__all__ = [
    "MCPClient",
]


@final
class MCPClient:
    @classmethod
    def stdio(
        cls,
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
                async with ClientSession(read, write) as session:
                    yield session

        return cls(mcp_stdio_session())

    @classmethod
    def sse(
        cls,
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

        return cls(mcp_sse_session())

    def __init__(
        self,
        session_manager: AbstractAsyncContextManager[ClientSession],
        /,
    ) -> None:
        self._session_manager: AbstractAsyncContextManager[ClientSession] = session_manager
        self._session: ClientSession

    async def resources_list(
        self,
        **extra: Any,
    ) -> Sequence[ResourceDeclaration]:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized throug async context entering"

        result: ListResourcesResult = await self._session.list_resources()
        # TODO: in theory there are some extra elements like pagination
        # we are not supporting it as for now (how to pass cursor anyways?)
        return [
            ResourceDeclaration(
                uri=resource.uri.unicode_string(),
                name=resource.name,
                description=resource.description,
                mime_type=resource.mimeType,
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
        ), "MCPClient has to be initialized throug async context entering"
        result: ReadResourceResult = await self._session.read_resource(uri=AnyUrl(uri))

        match [_convert_resource_content(element) for element in result.contents]:
            case [resource]:
                # if there is only a single element return it directly
                return resource

            case resources:
                return Resource(
                    uri=uri,
                    name="",  # TODO: resource name?
                    description=None,
                    content=resources,
                )

    async def prompts_list(
        self,
        **extra: Any,
    ) -> Sequence[PromptDeclaration]:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized throug async context entering"

        # TODO: in theory there are some extra elements like pagination
        # we are not supporting it as for now (how to pass cursor anyways?)
        return tuple(
            PromptDeclaration(
                name=prompt.name,
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
        ), "MCPClient has to be initialized throug async context entering"

        fetched_prompt: GetPromptResult = await self._session.get_prompt(
            name=name,
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
        )

    async def toolbox_fetch(
        self,
        *,
        suggest: bool | None = None,
        repeated_calls_limit: int | None = None,
        **extra: Any,
    ) -> Toolbox:
        tools: ListToolsResult = await self._session.list_tools()

        return Toolbox.of(
            *[_convert_tool(tool, self.tool_call) for tool in tools.tools],
            suggest=suggest,
            repeated_calls_limit=repeated_calls_limit,
        )

    async def tool_call(
        self,
        name: str,
        arguments: Mapping[str, BasicValue],
    ) -> MultimodalContent:
        # TODO: FIXME: linter
        result: CallToolResult = await self._session.call_tool(  # pyright: ignore
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

    async def __aenter__(self) -> tuple[ResourceRepository, PromptRepository, ExternalToolbox]:
        self._session = await self._session_manager.__aenter__()
        await self._session.initialize()

        return (
            ResourceRepository(
                list=self.resources_list,
                fetch=self.resource_fetch,
            ),
            PromptRepository(
                list=self.prompts_list,
                fetch=self.prompt_fetch,
            ),
            ExternalToolbox(
                fetch=self.toolbox_fetch,
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


def _convert_resource_content(
    resource: TextResourceContents | BlobResourceContents,
    /,
) -> Resource:
    match resource:
        case TextResourceContents():
            return Resource(
                uri=resource.uri.unicode_string(),
                name="",  # TODO: resource name?
                description=None,
                content=ResourceContent(
                    blob=resource.text.encode(),
                    mime_type=resource.mimeType or "text/plain",
                ),
            )

        case BlobResourceContents():
            return Resource(
                uri=resource.uri.unicode_string(),
                name="",  # TODO: resource name?
                description=None,
                content=ResourceContent(
                    blob=b64decode(resource.blob),
                    mime_type=resource.mimeType or "text/plain",
                ),
            )


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
                    media=validated_media_type(image.mimeType),
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
                                    media=validated_media_type(other),
                                )
                            )


def _convert_tool(
    mcp_tool: MCPTool,
    tool_call: Callable[[str, Mapping[str, BasicValue]], Coroutine[None, None, MultimodalContent]],
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
        specification=cast(ParametersSpecification, mcp_tool.inputSchema),
        function=remote_call,
        availability_check=None,
        format_result=_format_tool_result,
        format_failure=_format_tool_failure,
        direct_result=False,
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
