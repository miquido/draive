import json
from base64 import b64decode, b64encode
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from typing import Any, final

from haiway import Disposable, Disposables, State, as_dict, ctx
from mcp.server import NotificationOptions, Server, ServerRequestContext

# private module - the only route to the stream protocols in mcp 2.1, fragile across patches
from mcp.shared._stream_protocols import (  # pyright: ignore[reportPrivateUsage]
    ReadStream,
    WriteStream,
)
from mcp.shared.message import SessionMessage
from mcp.types import AudioContent as MCPAudioContent
from mcp.types import (
    BlobResourceContents,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
    TextResourceContents,
)
from mcp.types import ImageContent as MCPImageContent
from mcp.types import Resource as MCPResource
from mcp.types import ResourceTemplate as MCPResourceTemplate
from mcp.types import TextContent as MCPTextContent
from mcp.types import Tool as MCPTool
from starlette.types import ASGIApp

from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.resources import Resource, ResourceContent, ResourceReference, ResourceTemplate
from draive.tools import Tool, Toolbox

__all__ = ("MCPServer",)

type _LifespanState = Iterable[State]
type _RequestContext = ServerRequestContext[_LifespanState]
type _ListResourcesHandler = Callable[
    [_RequestContext, PaginatedRequestParams | None],
    Awaitable[ListResourcesResult],
]
type _ListResourceTemplatesHandler = Callable[
    [_RequestContext, PaginatedRequestParams | None],
    Awaitable[ListResourceTemplatesResult],
]
type _ReadResourceHandler = Callable[
    [_RequestContext, ReadResourceRequestParams],
    Awaitable[ReadResourceResult],
]
type _ListToolsHandler = Callable[
    [_RequestContext, PaginatedRequestParams | None],
    Awaitable[ListToolsResult],
]
type _CallToolHandler = Callable[
    [_RequestContext, CallToolRequestParams],
    Awaitable[CallToolResult],
]


@final
class MCPServer:
    __slots__ = ("_server",)

    def __init__(
        self,
        *,
        name: str,
        version: str | None = None,
        instructions: str | None = None,
        resources: Iterable[ResourceTemplate[Any] | Resource] = (),
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        disposables: Collection[Disposable] = (),
    ) -> None:
        @asynccontextmanager
        async def lifspan(server: Server) -> AsyncGenerator[Iterable[State]]:
            async with Disposables(disposables) as state:
                yield state

        # handlers are constructor arguments since mcp 2.0, they can't be registered after
        on_list_resources: _ListResourcesHandler | None = None
        on_list_resource_templates: _ListResourceTemplatesHandler | None = None
        on_read_resource: _ReadResourceHandler | None = None
        available_resources: Sequence[ResourceTemplate[Any] | Resource] = tuple(resources)
        if available_resources:
            (
                on_list_resources,
                on_list_resource_templates,
                on_read_resource,
            ) = self._resource_handlers(available_resources)

        on_list_tools: _ListToolsHandler | None = None
        on_call_tool: _CallToolHandler | None = None
        # an empty Toolbox is truthy, actual contents have to be verified
        toolbox: Toolbox = tools if isinstance(tools, Toolbox) else Toolbox.of(*tools)
        if toolbox.tools:
            on_list_tools, on_call_tool = self._tool_handlers(toolbox)

        self._server = Server[Iterable[State]](
            name=name,
            version=version if version is not None else "",
            instructions=instructions,
            lifespan=lifspan,
            on_list_resources=on_list_resources,
            on_list_resource_templates=on_list_resource_templates,
            on_read_resource=on_read_resource,
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
        )

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

    def prepare_streamable_http_asgi(
        self,
        *,
        path: str = "/mcp",
        json_response: bool = False,
        stateless: bool = False,
    ) -> ASGIApp:
        """Prepare an ASGI app serving the Streamable HTTP transport.

        This is the current remote transport, `prepare_asgi` (SSE) is deprecated
        by the protocol. Initialization options are derived from the server itself,
        the transport does not allow customizing them.
        """

        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Mount

        session_manager = StreamableHTTPSessionManager(
            app=self._server,
            json_response=json_response,
            stateless=stateless,
        )

        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncGenerator[None]:
            # the manager requires its own task group to be running
            async with session_manager.run():
                yield

        return Starlette(
            debug=__debug__,
            routes=[Mount(path, app=session_manager.handle_request)],
            lifespan=lifespan,
        )

    def prepare_asgi(
        self,
        notification_options: NotificationOptions | None = None,
        experimental_capabilities: Mapping[str, dict[str, Any]] | None = None,
    ) -> ASGIApp:
        """Prepare an ASGI app serving the deprecated SSE transport.

        Prefer `prepare_streamable_http_asgi` for new deployments.
        """

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
        read_stream: ReadStream[SessionMessage | Exception],
        write_stream: WriteStream[SessionMessage],
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

    def _resource_handlers(  # noqa: C901
        self,
        resources: Iterable[ResourceTemplate[...] | Resource],
        /,
    ) -> tuple[
        _ListResourcesHandler,
        _ListResourceTemplatesHandler | None,
        _ReadResourceHandler,
    ]:
        resource_declarations: list[MCPResource] = []
        resource_template_declarations: list[MCPResourceTemplate] = []
        available_resources: dict[str, ResourceTemplate[...] | Resource] = {}
        available_resource_templates: list[ResourceTemplate[...]] = []
        for resource in resources:
            match resource:
                case Resource():
                    available_resources[resource.uri] = resource
                    match resource.resource:
                        case ResourceContent() as content:
                            resource_declarations.append(
                                MCPResource(
                                    uri=resource.uri,
                                    mime_type=content.mime_type,
                                    name=resource.resource.meta.name or resource.uri,
                                    description=resource.resource.meta.description,
                                )
                            )

                        case _:
                            raise NotImplementedError(
                                "Multi-content resources are not supported yet"
                            )

                case ResourceTemplate():
                    if resource.has_args:
                        resource_template_declarations.append(
                            MCPResourceTemplate(
                                uri_template=resource.declaration.template_uri,
                                mime_type=resource.declaration.mime_type,
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
                                uri=resource.declaration.template_uri,
                                mime_type=resource.declaration.mime_type,
                                name=resource.declaration.meta.name
                                or resource.declaration.template_uri,
                                description=resource.declaration.meta.description,
                            )
                        )
                        # we treat it as a regular resource if template has no arguments
                        available_resources[resource.declaration.template_uri] = resource

        # the resources capability is derived from the `resources/list` handler registration,
        # it has to be always available to let clients discover resource templates as well
        async def list_resources(
            context: _RequestContext,
            params: PaginatedRequestParams | None,
        ) -> ListResourcesResult:
            async with ctx.scope(
                "list_resources",
                *context.lifespan_context,
            ):
                return ListResourcesResult(resources=resource_declarations)

        list_template_resources: _ListResourceTemplatesHandler | None = None
        if resource_template_declarations:

            async def handle_list_resource_templates(
                context: _RequestContext,
                params: PaginatedRequestParams | None,
            ) -> ListResourceTemplatesResult:
                async with ctx.scope(
                    "list_template_resources",
                    *context.lifespan_context,
                ):
                    return ListResourceTemplatesResult(
                        resource_templates=resource_template_declarations
                    )

            list_template_resources = handle_list_resource_templates

        async def read_resource(
            context: _RequestContext,
            params: ReadResourceRequestParams,
        ) -> ReadResourceResult:
            async with ctx.scope(
                "read_resource",
                *context.lifespan_context,
            ):
                resource: Resource
                uri_string: str = params.uri
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
                            raise ValueError(f"Resource '{uri_string}' is not defined")

                return ReadResourceResult(
                    contents=list(_resource_content(resource, uri=uri_string))
                )

        return (list_resources, list_template_resources, read_resource)

    def _tool_handlers(
        self,
        toolbox: Toolbox,
        /,
    ) -> tuple[_ListToolsHandler, _CallToolHandler]:
        async def list_tools(
            context: _RequestContext,
            params: PaginatedRequestParams | None,
        ) -> ListToolsResult:
            async with ctx.scope(
                "list_tools",
                *context.lifespan_context,
            ):
                return ListToolsResult(
                    tools=[
                        MCPTool(
                            name=tool.name,
                            description=tool.description,
                            input_schema=_json_schema(tool.parameters),
                        )
                        for tool in (tool.specification for tool in toolbox.tools.values())
                    ]
                )

        async def call_tool(
            context: _RequestContext,
            params: CallToolRequestParams,
        ) -> CallToolResult:
            async with ctx.scope(
                "call_tool",
                *context.lifespan_context,
            ):
                # the lowlevel server no longer converts exceptions to an error result
                try:
                    return CallToolResult(
                        content=list(
                            _convert_multimodal_content(
                                await toolbox.call(
                                    params.name,
                                    arguments=params.arguments or {},
                                )
                            )
                        ),
                        is_error=False,
                    )

                except Exception as exc:
                    ctx.log_error(f"MCP tool '{params.name}' call failed", exception=exc)
                    # the reason is reported back to let the model correct its call
                    return CallToolResult(
                        content=[
                            MCPTextContent(
                                type="text",
                                text=str(exc),
                            )
                        ],
                        is_error=True,
                    )

        return (list_tools, call_tool)


def _resource_content(
    resource: Resource,
    /,
    *,
    uri: str,
) -> Iterable[TextResourceContents | BlobResourceContents]:
    if isinstance(resource.resource, ResourceContent):
        match resource.resource.mime_type:
            case "text/plain" | "application/json":
                yield TextResourceContents(
                    uri=uri,
                    mime_type=resource.resource.mime_type,
                    text=b64decode(resource.resource.data).decode(),
                )

            case _:
                yield BlobResourceContents(
                    uri=uri,
                    mime_type=resource.resource.mime_type,
                    # blob carries the base64 payload, which is what we already hold
                    blob=resource.resource.data,
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
                        mime_type=mime,
                    )
                )

            elif mime.startswith("audio"):
                converted.append(
                    MCPAudioContent(
                        type="audio",
                        data=part.data,
                        mime_type=mime,
                    )
                )

            elif mime == "text/plain":
                converted.append(
                    MCPTextContent(
                        type="text",
                        text=b64decode(part.data).decode(),
                    )
                )

            elif mime == "application/json":
                encoded: str = part.data
                # a data URI stands in for the required uri field
                uri: str = f"data:{mime};base64,{encoded}"
                converted.append(
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri=uri,
                            mime_type=mime,
                            blob=encoded,
                        ),
                    )
                )

            else:
                # Unknown blob types: embed as a resource blob
                encoded: str = part.data
                uri: str = f"data:{mime};base64,{encoded}"
                converted.append(
                    EmbeddedResource(
                        type="resource",
                        resource=BlobResourceContents(
                            uri=uri,
                            mime_type=mime,
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
            encoded: str = b64encode(json.dumps(part.artifact).encode()).decode()
            uri: str = f"data:application/json;base64,{encoded}"
            converted.append(
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri=uri,
                        mime_type="application/json",
                        blob=encoded,
                    ),
                )
            )

    return converted


def _json_schema(
    schema: Mapping[str, Any] | None,
    /,
) -> dict[str, Any]:
    """Convert a schema to plain JSON types.

    The lowlevel server does not validate call arguments, however it serializes the schema
    with `model_dump(mode="json")` which does not handle nested haiway collections
    (`Map`, tuples) reliably.
    """

    if not schema:
        return {}

    return {key: _json_value(value) for key, value in schema.items()}


def _json_value(
    value: Any,
    /,
) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(element) for key, element in value.items()}  # pyright: ignore[reportUnknownVariableType]

    if isinstance(value, str | bytes):
        return value

    if isinstance(value, Sequence):
        return [_json_value(element) for element in value]  # pyright: ignore[reportUnknownVariableType]

    return value
