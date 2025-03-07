# Basic MCP

ModelContextProtocol (MCP) allows to easily extend application and LLM capabilities using standardized feature implementation. draive library comes with support for MCP both as a server and client allowing to build LLM based application even faster with more code reuse. Lets have a small example:

```python
from draive import (
    ConversationMessage,
    Toolbox,
    conversation_completion,
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
    OpenAI().lmm_invoking(),  # define used LMM to use OpenAI
    OpenAIChatConfig(model="gpt-4o-mini"),  # configure OpenAI model
    # prepare MCPClient, it will handle connection lifetime through context
    # and provide associated state with MCP functionalities
    disposables=[
        # we are going to use stdio connection with one of the example servers
        MCPClient.stdio( 
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/path/to/your/desktop",
            ],
        ),
    ]
):
    # request model using any appropriate method, i.e. conversation for chat
    response: ConversationMessage = await conversation_completion(
        # provide a prompt instruction
        instruction="You can access files on behalf of the user on their machine using available tools."
        " Desktop directory path is `/path/to/your/desktop`",
        # add user input
        input="What is on my desktop?",
        # define tools available to the model from MCP extensions
        tools=await Toolbox.external(),
    )
    print(response.content)
```