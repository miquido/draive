from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections.abc import AsyncGenerator, Collection, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, final
from uuid import uuid4

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from haiway import Disposable, Disposables, State, as_dict, ctx
from mcp.server import NotificationOptions, Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.message import SessionMessage
from mcp.types import AudioContent as MCPAudioContent
from mcp.types import BlobResourceContents, EmbeddedResource
from mcp.types import ImageContent as MCPImageContent
from mcp.types import Resource as MCPResource
from mcp.types import ResourceTemplate as MCPResourceTemplate
from mcp.types import TextContent as MCPTextContent
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl
from starlette.types import ASGIApp

from draive.models import Tool, Toolbox
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.resources import Resource, ResourceContent, ResourceReference, ResourceTemplate

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
        resources: Iterable[ResourceTemplate[Any] | Resource] | None = None,
        tools: Toolbox | Iterable[Tool] | None = None,
        disposables: Disposables | Collection[Disposable] | None = None,
    ) -> None:
        disposable: Disposables
        if disposables is None:
            disposable = Disposables(())

        elif isinstance(disposables, Disposables):
            disposable = disposables

        else:
            disposable = Disposables(disposables)

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

        if tools is not None:
            self._expose_tools(tools)

    async def run_stdio(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as streams:
            await self.run(
                read_stream=streams[0],
                write_stream=streams[1],
                notification_options=notification_options,
                experimental_capabilities=as_dict(experimental_capabilities),
            )

    def prepare_asgi(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: Mapping[str, dict[str, Any]] | None = None,
    ) -> ASGIApp:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Any):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await self.run(
                    read_stream=streams[0],
                    write_stream=streams[1],
                    notification_options=notification_options,
                    experimental_capabilities=as_dict(experimental_capabilities),
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
        resources: Iterable[ResourceTemplate[...] | Resource],
        /,
    ) -> None:
        resource_declarations: list[MCPResource] = []
        resource_template_declarations: list[MCPResourceTemplate] = []
        available_resources: dict[str, ResourceTemplate[...] | Resource] = {}
        available_resource_templates: list[ResourceTemplate[...]] = []
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
                                    name=resource.meta.name or resource.uri,
                                    description=resource.meta.description,
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
                                uriTemplate=resource.declaration.template_uri,
                                mimeType=resource.declaration.mime_type,
                                name=resource.declaration.meta.name
                                or resource.declaration.template_uri,
                                description=resource.declaration.meta.description,
                            )
                        )
                        # TODO: we might need to sort based on template uri matching priorities
                        available_resource_templates.append(resource)

                    else:
                        resource_declarations.append(
                            MCPResource(
                                uri=AnyUrl(resource.declaration.template_uri),
                                mimeType=resource.declaration.mime_type,
                                name=resource.declaration.meta.name
                                or resource.declaration.template_uri,
                                description=resource.declaration.meta.description,
                            )
                        )
                        # we treat it as a regular resource if template has no arguments
                        available_resources[resource.declaration.template_uri] = resource

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
                        name=tool.name,
                        description=tool.description,
                        inputSchema=as_dict(tool.parameters) or {},
                    )
                    for tool in (tool.specification for tool in toolbox.tools.values())
                ]

        @self._server.call_tool()
        async def call_tool(  # pyright: ignore[reportUnusedFunction]
            name: str,
            arguments: Mapping[str, Any],
        ) -> Sequence[MCPTextContent | MCPImageContent | MCPAudioContent | EmbeddedResource]:
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
    if isinstance(resource.content, ResourceContent):
        match resource.content.mime_type:
            case "text/plain" | "application/json":
                yield ReadResourceContents(
                    content=urlsafe_b64decode(resource.content.data).decode(),
                    mime_type=resource.content.mime_type,
                )

            case _:
                yield ReadResourceContents(
                    content=urlsafe_b64decode(resource.content.data),
                    mime_type=resource.content.mime_type,
                )

    else:
        # Multi-content resources (lists of references) are not supported for server reads
        raise NotImplementedError("Multi-content resources are not supported yet")


def _convert_multimodal_content(
    content: MultimodalContent,
) -> Sequence[MCPTextContent | MCPImageContent | MCPAudioContent | EmbeddedResource]:
    converted: list[MCPTextContent | MCPImageContent | MCPAudioContent | EmbeddedResource] = []
    for part in content.parts:
        if isinstance(part, TextContent):
            converted.append(
                MCPTextContent(
                    type="text",
                    text=part.text,
                )
            )

        elif isinstance(part, ResourceContent):
            mime: str = part.mime_type
            if mime.startswith("image"):
                converted.append(
                    MCPImageContent(
                        type="image",
                        data=part.data,
                        mimeType=mime,
                    )
                )

            elif mime.startswith("audio"):
                converted.append(
                    MCPAudioContent(
                        type="audio",
                        data=part.data,
                        mimeType=mime,
                    )
                )

            elif mime == "text/plain":
                converted.append(
                    MCPTextContent(
                        type="text",
                        text=urlsafe_b64decode(part.data).decode(),
                    )
                )

            elif mime == "application/json":
                encoded: str = part.data
                # Provide a data URI to satisfy required AnyUrl
                uri = AnyUrl(f"data:{mime};base64,{encoded}")
                converted.append(
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri=uri,
                            mimeType=mime,
                            blob=encoded,
                        ),
                    )
                )

            else:
                # Unknown blob types: embed as a resource blob
                encoded: str = part.data
                uri = AnyUrl(f"data:{mime};base64,{encoded}")
                converted.append(
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri=uri,
                            mimeType=mime,
                            blob=encoded,
                        ),
                    )
                )

        elif isinstance(part, ResourceReference):
            # We don't return links yet; ask callers to provide content
            # we could try to resolve those contextually using ResourceRepository
            raise NotImplementedError(
                "MCP resource links are not supported yet; provide content blobs instead"
            )

        else:
            assert isinstance(part, ArtifactContent)  # nosec: B101
            encoded: str = urlsafe_b64encode(part.artifact.to_json().encode()).decode()
            uri = AnyUrl(f"data:application/json;base64,{encoded}")
            converted.append(
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri=uri,
                        mimeType="application/json",
                        blob=encoded,
                    ),
                )
            )

    return converted
