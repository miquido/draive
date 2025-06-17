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
from draive import ConversationMessage, Conversation, ctx
from draive.openai import OpenAIChatConfig, OpenAI

# initialize dependencies and configuration
async with ctx.scope(
    "basics",
    OpenAIChatConfig(model="gpt-3.5-turbo-0125"),  # configure OpenAI model
    disposables=(OpenAI(),),  # specify OpenAI as the LMM resource
):
    # request conversation completion
    response: ConversationMessage = await Conversation.completion(
        # provide a prompt instruction
        instruction="You are a helpful assistant.",
        # add user input
        input="Hi! What is the time now?",
        # define tools available to the model
        tools=[utc_datetime],
    )
    print(response)
```

    identifier: dd2a86730a3441939359e960f0cc2da3
    role: model
    author: None
    created: 2025-03-07 12:40:30.130777+00:00
    content:
      parts:
        - text: The current UTC time and date is Friday, 7th March 2025, 12:40:29.
          meta: None
    meta: None
