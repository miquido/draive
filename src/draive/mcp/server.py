from base64 import urlsafe_b64encode
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, final
from uuid import uuid4

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from haiway import Disposable, Disposables, State, as_dict, ctx
from mcp.server import NotificationOptions, Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.message import SessionMessage
from mcp.types import EmbeddedResource, GetPromptResult
from mcp.types import ImageContent as MCPImageContent
from mcp.types import Prompt as MCPPrompt
from mcp.types import PromptArgument as MCPPromptArgument
from mcp.types import PromptMessage as MCPPromptMessage
from mcp.types import Resource as MCPResource
from mcp.types import ResourceTemplate as MCPResourceTemplate
from mcp.types import TextContent as MCPTextContent
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl
from starlette.types import ASGIApp

from draive.lmm import LMMCompletion, LMMContextElement, LMMInput
from draive.multimodal import MediaData, MediaReference, MultimodalContent, TextContent
from draive.prompts import Prompt, PromptTemplate
from draive.resources import Resource, ResourceContent, ResourceTemplate
from draive.tools import Tool, Toolbox

__all__ = ("MCPServer",)


@final
class MCPServer:
    __slots__ = ("_server",)

    def __init__(
        self,
        *,
        name: str,
        version: str | None = None,
        instructions: str | None = None,
        resources: Iterable[ResourceTemplate | Resource] | None = None,
        prompts: Iterable[PromptTemplate[Any] | Prompt] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        disposables: Disposables | Iterable[Disposable] | None = None,
    ) -> None:
        disposable: Disposables
        match disposables:
            case None:
                disposable = Disposables()

            case Disposables() as prepared:
                disposable = prepared

            case other:
                disposable = Disposables(*other)

        @asynccontextmanager
        async def lifspan(server: Server) -> AsyncGenerator[Iterable[State]]:
            state: Iterable[State] = await disposable.prepare()
            try:
                yield state

            finally:
                await disposable.dispose()

        self._server = Server[Iterable[State]](
            name=name,
            version=version,
            instructions=instructions,
            lifespan=lifspan,
        )

        if resources is not None:
            self._expose_resources(resources)

        if prompts is not None:
            self._expose_prompts(prompts)

        if tools is not None:
            self._expose_tools(tools)

    async def run_stdio(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as streams:
            await self.run(
                read_stream=streams[0],
                write_stream=streams[1],
                notification_options=notification_options,
                experimental_capabilities=experimental_capabilities,
            )

    def prepare_asgi(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> ASGIApp:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await self.run(
                    read_stream=streams[0],
                    write_stream=streams[1],
                    notification_options=notification_options,
                    experimental_capabilities=experimental_capabilities,
                )

        return Starlette(
            debug=__debug__,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

    async def run(
        self,
        read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
        write_stream: MemoryObjectSendStream[SessionMessage],
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        await self._server.run(
            read_stream=read_stream,
            write_stream=write_stream,
            initialization_options=self._server.create_initialization_options(
                notification_options=notification_options,
                experimental_capabilities=experimental_capabilities,
            ),
            raise_exceptions=False,
        )

    def _expose_resources(  # noqa: C901
        self,
        resources: Iterable[ResourceTemplate | Resource],
        /,
    ) -> None:
        resource_declarations: list[MCPResource] = []
        resource_template_declarations: list[MCPResourceTemplate] = []
        available_resources: dict[str, ResourceTemplate | Resource] = {}
        available_resource_templates: list[ResourceTemplate] = []
        for resource in resources:
            match resource:
                case Resource():
                    available_resources[resource.uri] = resource
                    match resource.content:
                        case ResourceContent() as content:
                            resource_declarations.append(
                                MCPResource(
                                    uri=AnyUrl(resource.uri),
                                    mimeType=content.mime_type,
                                    name=resource.name,
                                    description=resource.description,
                                )
                            )

                        case _:
                            raise NotImplementedError(
                                "Multi-content resources are not supported yet"
                            )

                case ResourceTemplate():
                    if resource.arguments:
                        resource_template_declarations.append(
                            MCPResourceTemplate(
                                uriTemplate=resource.declaration.uri_template,
                                mimeType=resource.declaration.mime_type,
                                name=resource.declaration.name,
                                description=resource.declaration.description,
                            )
                        )
                        # TODO: we might need to sort based on template uri matching priorities
                        available_resource_templates.append(resource)

                    else:
                        resource_declarations.append(
                            MCPResource(
                                uri=AnyUrl(resource.declaration.uri_template),
                                mimeType=resource.declaration.mime_type,
                                name=resource.declaration.name,
                                description=resource.declaration.description,
                            )
                        )
                        # we treat it as a regular resource if template has no arguments
                        available_resources[resource.declaration.uri_template] = resource

        if resource_declarations:

            @self._server.list_resources()
            async def list_resources() -> list[MCPResource]:  # pyright: ignore[reportUnusedFunction]
                async with ctx.scope(
                    "list_resources",
                    *self._server.request_context.lifespan_context,
                ):
                    return resource_declarations

        if resource_template_declarations:

            @self._server.list_resource_templates()
            async def list_template_resources() -> list[MCPResourceTemplate]:  # pyright: ignore[reportUnusedFunction]
                async with ctx.scope(
                    "list_template_resources",
                    *self._server.request_context.lifespan_context,
                ):
                    return resource_template_declarations

        @self._server.read_resource()
        async def read_resource(uri: AnyUrl) -> Iterable[ReadResourceContents]:  # pyright: ignore[reportUnusedFunction]
            async with ctx.scope(
                "read_resource",
                *self._server.request_context.lifespan_context,
            ):
                resource: Resource
                uri_string: str = uri.unicode_string()
                # First check for exact match in available_resources
                match available_resources.get(uri_string):
                    case Resource() as available_resource:
                        resource = available_resource

                    case ResourceTemplate() as resource_template:
                        resource = await resource_template.resolve_from_uri(uri_string)

                    case None:
                        # if there is no exact match check in templates
                        for template in available_resource_templates:
                            if template.matches_uri(uri_string):
                                resource = await template.resolve_from_uri(uri_string)
                                break

                        else:
                            raise ValueError(f"Resource '{uri}' is not defined")

                return _resource_content(resource)

    def _expose_prompts(
        self,
        prompts: Iterable[PromptTemplate[Any] | Prompt],
        /,
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

        @self._server.list_prompts()
        async def list_prompts() -> list[MCPPrompt]:  # pyright: ignore[reportUnusedFunction]
            async with ctx.scope(
                "list_prompts",
                *self._server.request_context.lifespan_context,
            ):
                return prompt_declarations

        @self._server.get_prompt()
        async def get_prompt(  # pyright: ignore[reportUnusedFunction]
            name: str,
            arguments: Mapping[str, Any] | None,
        ) -> GetPromptResult:
            async with ctx.scope(
                "get_prompt",
                *self._server.request_context.lifespan_context,
            ):
                match available_prompts.get(name):
                    case None:
                        raise ValueError(f"Prompt '{name}' is not defined")

                    case Prompt() as direct_prompt:
                        return GetPromptResult(
                            description=direct_prompt.description,
                            messages=[
                                _convert_message(element) for element in direct_prompt.content
                            ],
                        )

                    case PromptTemplate() as dynamic_prompt:
                        resolved_prompt: Prompt = await dynamic_prompt.resolve(arguments or {})
                        return GetPromptResult(
                            description=resolved_prompt.description,
                            messages=[
                                _convert_message(element) for element in resolved_prompt.content
                            ],
                        )

    def _expose_tools(
        self,
        tools: Toolbox | Iterable[Tool],
        /,
    ) -> None:
        toolbox: Toolbox
        match tools:
            case Toolbox() as tools:
                toolbox = tools

            case tools:
                toolbox = Toolbox.of(*tools)

        @self._server.list_tools()
        async def list_tools() -> list[MCPTool]:  # pyright: ignore[reportUnusedFunction]
            async with ctx.scope(
                "list_tools",
                *self._server.request_context.lifespan_context,
            ):
                return [
                    MCPTool(
                        name=tool["name"],
                        description=tool["description"],
                        inputSchema=as_dict(tool["parameters"]) or {},
                    )
                    for tool in (
                        tool.specification for tool in toolbox.tools.values() if tool.available
                    )
                ]

        @self._server.call_tool()
        async def call_tool(  # pyright: ignore[reportUnusedFunction]
            name: str,
            arguments: Mapping[str, Any],
        ) -> Sequence[MCPTextContent | MCPImageContent | EmbeddedResource]:
            async with ctx.scope(
                "call_tool",
                *self._server.request_context.lifespan_context,
            ):
                return _convert_multimodal_content(
                    await toolbox.call_tool(
                        name,
                        call_id=uuid4().hex,
                        arguments=arguments,
                    )
                )


def _resource_content(
    resource: Resource,
) -> Iterable[ReadResourceContents]:
    match resource.content:
        case ResourceContent() as content:
            match content.mime_type:
                case "text/plain" | "application/json":
                    yield ReadResourceContents(
                        content=content.blob.decode(),
                        mime_type=content.mime_type,
                    )

                case _:
                    yield ReadResourceContents(
                        content=content.blob,
                        mime_type=content.mime_type,
                    )

        case [*contents]:  # is it intended use of nested resource contents?
            for content in contents:
                yield from _resource_content(content)


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

            case MediaData() as media_data:
                if media_data.kind != "image":
                    raise NotImplementedError(
                        f"Media support for {media_data.media} is not implemented"
                    )

                converted.append(
                    MCPImageContent(
                        type="image",
                        data=urlsafe_b64encode(media_data.data).decode(),
                        mimeType=media_data.media,
                    )
                )

            case MediaReference():
                # TODO: download images on the fly?
                raise NotImplementedError(
                    "Media reference support is not implemented, please use data blobs instead"
                )

            case other:
                converted.append(
                    MCPTextContent(
                        type="text",
                        text=other.to_json(),
                    )
                )

    return converted
