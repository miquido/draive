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

## Example: Remote MCP Server

For remote servers use the Streamable HTTP transport. `MCPClient.sse` is still available but the
SSE transport is deprecated by the protocol, prefer Streamable HTTP for new integrations.

```python
from draive.mcp import MCPClient

MCPClient.streamable_http(
    url="https://example.com/mcp",
    headers={"Authorization": "Bearer <token>"},
)
```

## Example: Parallel Web Search

[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) exposes `web_search`
and `web_fetch` over Streamable HTTP. Its anonymous tier is free, requires no API key, and uses
Fast search mode with rate limits. This example calls the search tool directly, so it doesn't
need a model provider or inference credentials.

Install the MCP extra in a Python 3.14+ environment:

```bash
pip install 'draive[mcp]'
```

Save the following as `parallel_search.py` and run `python parallel_search.py`:

```python
import asyncio
from importlib.metadata import version
from uuid import uuid4

from draive import ToolsProvider, ctx
from draive.mcp import MCPClient


async def main() -> None:
    session_id = uuid4().hex
    async with ctx.scope(
        "parallel-search",
        disposables=(
            MCPClient.streamable_http(
                "parallel-search",
                url="https://search.parallel.ai/mcp",
                headers={"User-Agent": f"draive/{version('draive')}"},
                features=(ToolsProvider,),
            ),
        ),
    ):
        toolbox = await ToolsProvider.toolbox()
        result = await toolbox.call(
            "web_search",
            arguments={
                "objective": "Find the Python documentation for asyncio TaskGroup.",
                "search_queries": ["Python asyncio TaskGroup documentation"],
                "session_id": session_id,
            },
        )
        print(result.to_str())


if __name__ == "__main__":
    asyncio.run(main())
```

The output contains search results with source URLs and excerpts. Reuse `session_id` for later
`web_search` or `web_fetch` calls in the same conversation. The client discovers tools through
`ToolsProvider` and converts their results to Draive's `MultimodalContent`; leaving the scope closes
the connection. No environment files or Parallel credentials are loaded. Add the toolbox to
`Conversation.completion(..., tools=toolbox)` inside the same scope if you want a configured model
provider to use these tools; that provider still needs its own setup and may incur inference costs.
Existing provider configurations stay unchanged unless you explicitly add this client to a scope.

## Exposing Draive Tools

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
