# Basic MCP

Use MCP servers to expose external tools/resources to Draive through `MCPClient`.

This pattern is useful when you want models to call capabilities provided by external MCP servers
without writing custom SDK integration code.

## Example: Filesystem MCP Server

```python
from draive import Conversation, ToolsProvider, ctx, load_env, setup_logging
from draive.mcp import MCPClient
from draive.openai import OpenAI, OpenAIResponsesConfig

load_env()
setup_logging("mcp")

async with ctx.scope(
    "mcp",
    OpenAIResponsesConfig(model="gpt-5.5"),
    disposables=(
        OpenAI(),
        # Start MCP stdio transport and register tool/resource states in context.
        MCPClient.stdio(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/Users/myname/checkmeout",
            ],
        ),
    ),
):
    # Build toolbox from currently available MCP tools.
    toolbox = await ToolsProvider.toolbox(suggesting=True)

    stream = Conversation.completion(
        instructions=(
            "You can access user files using available tools. "
            "Directory path is /Users/myname/checkmeout."
        ),
        message="What files are in checkmeout directory?",
        tools=toolbox,
    )

    async for chunk in stream:
        print(chunk)
```

`MCPClient` contributes `ToolsProvider`/`ResourcesRepository` states inside `ctx.scope(...)`, so you
can load MCP tools dynamically and keep all dependencies lifecycle-managed by context.

## Tool Argument Schemas

The input schema of an MCP tool is translated into the typed `ModelToolParametersSpecification`
every Draive tool declares, so all providers receive the same well-formed schema:

- References to `$defs` and `definitions` are inlined, a recursive definition is anchored and
    referenced as `#Name`, the way a recursive `State` is.
- A reference with sibling keywords and the branches of `allOf` are combined into one schema,
    `oneOf` becomes `anyOf` and `const` a single value `enum`.
- Arguments declared by root `anyOf`/`oneOf` branches become arguments of the tool: required are
    only the ones every branch requires, and a property declared differently by the branches keeps
    each declaration as an `anyOf`.
- Objects with properties are closed for undeclared keys. Keywords which only constrain values
    further (`default`, `minimum`, `pattern`, `title`...) are dropped, the server still validates
    them when the tool is called.

A tool whose schema cannot be translated without changing what it accepts, for example one with an
unresolvable `$ref` or an argument of any type (`{}`), is skipped with a warning naming the tool and
the reason; the other tools of the server stay available.

## Example: Remote MCP Server

For remote servers use the Streamable HTTP transport. `MCPClient.sse` is still available but the SSE
transport is deprecated by the protocol, prefer Streamable HTTP for new integrations.

```python
from draive.mcp import MCPClient

MCPClient.streamable_http(
    url="https://example.com/mcp",
    headers={"Authorization": "Bearer <token>"},
)
```

Exposing Draive tools and resources the other way around works through `MCPServer`, which serves
either stdio or an ASGI app:

```python
from draive.mcp import MCPServer

server = MCPServer(name="my-server", version="1.0.0", tools=[my_tool])

# stdio for local subprocess usage
await server.run_stdio()

# or an ASGI app to serve over Streamable HTTP
app = server.prepare_streamable_http_asgi(path="/mcp")
```
