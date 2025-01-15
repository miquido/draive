from base64 import b64encode
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from uuid import uuid4

from haiway import as_dict
from mcp.server import Server
from mcp.types import EmbeddedResource, GetPromptResult
from mcp.types import ImageContent as MCPImageContent
from mcp.types import Prompt as MCPPrompt
from mcp.types import PromptArgument as MCPPromptArgument
from mcp.types import PromptMessage as MCPPromptMessage
from mcp.types import TextContent as MCPTextContent
from mcp.types import Tool as MCPTool

from draive.lmm import LMMCompletion, LMMContextElement, LMMInput
from draive.multimodal import MediaContent, MultimodalContent, TextContent
from draive.prompts import Prompt, PromptTemplate
from draive.tools import AnyTool, Toolbox

__all__ = [
    "expose_prompts",
    "expose_tools",
]


def expose_prompts(
    prompts: Iterable[PromptTemplate[Any] | Prompt],
    /,
    server: Server,
) -> None:
    prompt_declarations: list[MCPPrompt] = []
    available_prompts: dict[str, PromptTemplate[Any] | Prompt] = {}
    for prompt in prompts:
        match prompt:
            case Prompt():
                prompt_declarations.append(
                    MCPPrompt(
                        name=prompt.name,
                        arguments=None,
                        description=prompt.description,
                    )
                )
                available_prompts[prompt.name] = prompt

            case PromptTemplate():
                prompt_declarations.append(
                    MCPPrompt(
                        name=prompt.declaration.name,
                        arguments=[
                            MCPPromptArgument(
                                name=argument.name,
                                description=argument.specification.get("description"),
                                required=argument.required,
                            )
                            for argument in prompt.declaration.arguments
                        ],
                        description=prompt.declaration.description,
                    )
                )
                available_prompts[prompt.declaration.name] = prompt

    @server.list_prompts()
    async def list_prompts() -> list[MCPPrompt]:  # pyright: ignore[reportUnusedFunction]
        return prompt_declarations

    @server.get_prompt()
    async def get_prompt(  # pyright: ignore[reportUnusedFunction]
        name: str,
        arguments: Mapping[str, Any] | None,
    ) -> GetPromptResult:
        match available_prompts.get(name):
            case None:
                raise ValueError(f"Prompt '{name}' is not defined")

            case Prompt() as direct_prompt:
                return GetPromptResult(
                    description=direct_prompt.description,
                    messages=[_convert_message(element) for element in direct_prompt.content],
                )

            case PromptTemplate() as dynamic_prompt:
                resolved_prompt: Prompt = await dynamic_prompt.resolve(arguments or {})
                return GetPromptResult(
                    description=resolved_prompt.description,
                    messages=[_convert_message(element) for element in resolved_prompt.content],
                )


def _convert_message(
    element: LMMContextElement,
    /,
) -> MCPPromptMessage:
    match element:
        case LMMInput():
            match _convert_multimodal_content(element.content):
                case [content]:
                    return MCPPromptMessage(
                        role="user",
                        content=content,
                    )
                case []:
                    return MCPPromptMessage(
                        role="user",
                        # convert empty content to empty text to prevent errors
                        content=MCPTextContent(
                            type="text",
                            text="",
                        ),
                    )

                case [*_]:
                    raise NotImplementedError(
                        f"Unsuppoprted MCP prompt element content: {element.content}!"
                    )

        case LMMCompletion():
            match _convert_multimodal_content(element.content):
                case [content]:
                    return MCPPromptMessage(
                        role="assistant",
                        content=content,
                    )
                case []:
                    return MCPPromptMessage(
                        role="assistant",
                        # convert empty content to empty text to prevent errors
                        content=MCPTextContent(
                            type="text",
                            text="",
                        ),
                    )

                case [*_]:
                    raise NotImplementedError(
                        f"Unsuppoprted MCP prompt element content: {element.content}!"
                    )

        case _:
            raise NotImplementedError(f"Unsuppoprted MCP prompt context element: {type(element)}!")


def expose_tools(
    tools: Toolbox | Iterable[AnyTool],
    /,
    server: Server,
) -> None:
    toolbox: Toolbox = Toolbox.out_of(tools)

    @server.list_tools()
    async def list_tools() -> list[MCPTool]:  # pyright: ignore[reportUnusedFunction]
        return [
            MCPTool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=as_dict(tool["parameters"]),
            )
            for tool in toolbox.available_tools()
        ]

    @server.call_tool()
    async def call_tool(  # pyright: ignore[reportUnusedFunction]
        name: str,
        arguments: Mapping[str, Any],
    ) -> Sequence[MCPTextContent | MCPImageContent | EmbeddedResource]:
        return _convert_multimodal_content(
            await toolbox.call_tool(
                name,
                call_id=uuid4().hex,
                arguments=arguments,
            )
        )


def _convert_multimodal_content(
    content: MultimodalContent,
) -> Sequence[MCPTextContent | MCPImageContent | EmbeddedResource]:
    converted: list[MCPTextContent | MCPImageContent | EmbeddedResource] = []
    for part in content.parts:
        match part:
            case TextContent() as text:
                converted.append(
                    MCPTextContent(
                        type="text",
                        text=text.text,
                    )
                )

            case MediaContent() as media:
                if media.kind != "image":
                    raise NotImplementedError(f"Media support for {media.media} is not implemented")

                if not isinstance(media.source, bytes):
                    # TODO: download images on the fly?
                    raise NotImplementedError(
                        "Media support for urls is not implemented, please use data blobs instead"
                    )

                converted.append(
                    MCPImageContent(
                        type="image",
                        data=b64encode(media.source).decode(),
                        mimeType=media.media,
                    )
                )

            case other:
                converted.append(
                    MCPTextContent(
                        type="text",
                        text=other.as_json(),
                    )
                )

    return converted
