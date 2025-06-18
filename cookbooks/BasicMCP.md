# Basic MCP

ModelContextProtocol (MCP) allows to easily extend application and LLM capabilities using standardized feature implementation. Draive library comes with support for MCP both as a server and client allowing to build LLM based application even faster with more code reuse.

For this example we've created a directory in our home folder and its structure looks like this:
```
checkmeout/
    .. file1
    .. file10
    .. file2
    .. file3
```

```python
from draive import (
    ConversationMessage,
    Toolbox,
    Conversation,
    ctx,
    load_env,
    setup_logging,
)
from draive.mcp import MCPClient
from draive.openai import OpenAIChatConfig, OpenAI

load_env() # load .env variables
setup_logging("mcp")


# initialize dependencies and configuration
async with ctx.scope(
    "mcp",
    OpenAIChatConfig(model="gpt-4o-mini"),  # configure OpenAI model
    # prepare MCPClient, it will handle connection lifetime through context
    # and provide associated state with MCP functionalities
    disposables=(
        OpenAI(),  # specify OpenAI as the LMM resource
        # we are going to use stdio connection with one of the example servers
        MCPClient.stdio(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/Users/myname/checkmeout",
            ],
        ),
    )
):
    # request model using any appropriate method, i.e. conversation for chat
    response: ConversationMessage = await Conversation.completion(
        # provide a prompt instruction
        instruction="You can access files on behalf of the user on their machine using available tools."
        " Desktop directory path is `/Users/myname/checkmeout`",
        # add user input
        input="What files are in checkmeout directory?",
        # define tools available to the model from MCP extensions
        tools=await Toolbox.fetched(),
    )
    print(response.content)
```
    The `checkmeout` directory contains the following files:

    - file1
    - file10
    - file2
    - file3
