from asyncio import gather
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, cast, final

from haiway import as_dict
from mcp import ClientSession, GetPromptResult, ListToolsResult
from mcp import Tool as MCPTool
from mcp.types import CallToolResult, EmbeddedResource, ImageContent, TextContent

from draive.lmm import LMMCompletion, LMMContextElement, LMMInput
from draive.lmm.types import LMMToolError
from draive.multimodal import MediaContent, MultimodalContent, validated_media_type
from draive.parameters import BasicValue, ParametersSpecification
from draive.prompts import Prompt, PromptDeclaration, PromptDeclarationArgument, PromptRepository
from draive.tools import AnyTool, ExternalToolbox, Tool, Toolbox

__all__ = [
    "MCPClient",
]


@final
class MCPClient:
    def __init__(
        self,
        session_manager: AbstractAsyncContextManager[ClientSession],
        /,
    ) -> None:
        self._session_manager: AbstractAsyncContextManager[ClientSession] = session_manager
        self._session: ClientSession

    async def prompts_list(
        self,
    ) -> Sequence[PromptDeclaration]:
        assert hasattr(  # nosec: B101
            self,
            "_session",
        ), "MCPClient has to be initialized throug async context entering"

        return tuple(
            PromptDeclaration(
                name=prompt.name,
                description=prompt.description,
                arguments=tuple(
                    PromptDeclarationArgument(
                        name=argument.name,
                        description=argument.description,
                        required=argument.required if argument.required is not None else True,
                    )
                    for argument in prompt.arguments
                )
                if prompt.arguments
                else (),
            )
            # TODO: in theory there are some extra elements like pagination
            # we are not supporting it as for now (how to pass cursor anyways?)
            for prompt in (await self._session.list_prompts()).prompts
        )

    async def prompt_fetch(
        self,
        name: str,
        *,
        arguments: Mapping[str, str] | None,
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
        extending: Toolbox | Iterable[AnyTool] | None,
    ) -> Toolbox:
        tools: ListToolsResult = await self._session.list_tools()

        # TODO: FIXME: linter
        return Toolbox.out_of([_convert_tool(tool, self.tool_call) for tool in tools.tools])  # pyright: ignore

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

    async def __aenter__(self) -> tuple[PromptRepository, ExternalToolbox]:
        self._session = await self._session_manager.__aenter__()
        await self._session.initialize()

        return (
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


async def _convert_content(
    content: TextContent | ImageContent | EmbeddedResource,
    /,
) -> MultimodalContent:
    match content:
        case TextContent() as text:
            return MultimodalContent.of(text.text)

        case ImageContent() as image:
            return MultimodalContent.of(
                MediaContent.base64(
                    image.data,
                    media=validated_media_type(image.mimeType),
                )
            )

        case other:
            raise NotImplementedError(f"Unsuppoprted MCP prompt message element: {type(other)}!")


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
