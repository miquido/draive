## Basic usage of draive with OpenAI

Add OPENAI_API_KEY key to the .env file to allow access to OpenAI services.
```python
from draive import load_env

load_env()  # loads OPENAI_API_KEY from .env file
```
```python
from datetime import UTC, datetime

from draive import tool


# prepare a basic tool for getting current date and time
@tool(description="UTC time and date now")
async def utc_datetime() -> str:
    return datetime.now(UTC).strftime("%A %d %B, %Y, %H:%M:%S")
```
```python
from draive import ConversationMessage, conversation_completion, ctx
from draive.openai import OpenAIChatConfig, OpenAI

# initialize dependencies and configuration
async with ctx.scope(
    "basics",
    OpenAI().lmm_invoking(),  # define used LMM to use OpenAI
    OpenAIChatConfig(model="gpt-3.5-turbo-0125"),  # configure OpenAI model
):
    # request conversation completion
    response: ConversationMessage = await conversation_completion(
        # provide a prompt instruction
        instruction="You are a helpful assistant.",
        # add user input
        input="Hi! What is the time now?",
        # define tools available to the model
        tools=[utc_datetime],
    )
    print(response)
```