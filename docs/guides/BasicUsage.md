# Basic LLM usage

Draive framework provides various ways to use LLM depending on the use case. The simplest interface
is to generate text by using `TextGeneration.generate` method. We can use it to make a simple text
completion function.

```python
from draive import TextGeneration

async def text_completion(text: str) -> str:
    # TextGeneration.generate is a simple interface for generating text
    return await TextGeneration.generate(
        # We have to provide instructions / system prompt to instruct the model
        instructions="Prepare the simplest completion of a given text",
        # input is provided separately
        input=text,
    )
```

The result of this function is a completion from a currently used model. What is a currently used
model? We have to define it yet by providing basic setup of state and dependencies. In this example
we are going to use OpenAI client, you have to provide the api key to that service in .env file with
`OPENAI_API_KEY` key before running.

```python
from draive import load_env

load_env()  # load .env variables
```

When we have .env loaded we can prepare a context scope with the OpenAI client and use our function.
The low-level model interface is the GenerativeModel. Draive supports multi-model solutions out of
the box. Selecting the OpenAI client and configuration in the context chooses this provider for our
generations.

```python
from draive import ctx
from draive.openai import OpenAIResponsesConfig, OpenAI

async with ctx.scope(  # prepare new context
    "basics",
    OpenAIResponsesConfig(model="gpt-5.5"),  # select used model
    disposables=(OpenAI(),),  # initialize OpenAI client
):
    result: str = await text_completion(
        text="Roses are red...",
    )

    print(result)
```

```text
Violets are blue,
Sugar is sweet,
And so are you.
```

As we now know how to set up OpenAI as our model provider, we can start customizing it further by
providing model configuration.

```python
from draive.openai import OpenAIResponsesConfig

async with ctx.scope(  # prepare the new context
    "basics",
    # define GPT model configuration as a context scope state
    OpenAIResponsesConfig(
        model="gpt-5.5",
        reasoning="none",
    ),
    disposables=(OpenAI(),),
):
    # now we are using gpt-5.5 with reasoning disabled
    result: str = await text_completion(
        text="Roses are red...",
    )

    print("RESULT gpt-5.5 | reasoning none:", result)

    # we can update the configuration to change any parameter for nested context
    with ctx.updating(
        # we are updating the current context value instead of making a new one
        # this allows to preserve other elements of the configuration
        ctx.state(OpenAIResponsesConfig).updating(
            model="gpt-5.6-terra",
        ),
    ):
        # now we are using gpt-5.6-terra with reasoning disabled
        result = await text_completion(
            text="Roses are red...",
        )

        print("RESULT gpt-5.6-terra | reasoning none:", result)

    # we can also update the configuration for a single call
    # by passing it explicitly to the provider
    result = await TextGeneration.generate(
        instructions="Prepare simplest completion of given text",
        input="Roses are red...",
        config=ctx.state(OpenAIResponsesConfig).updating(reasoning="low"),
    )

    print("RESULT gpt-5.5 | reasoning low:", result)
```

```text
RESULT gpt-5.5 | reasoning none: Violets are blue.

RESULT gpt-5.6-terra | reasoning none: Violets are blue.

RESULT gpt-5.5 | reasoning low: Violets are blue.
```

Since we know the basics, now we can examine the details of our execution to see what actually
happened inside. We can setup the logger before execution and assign a logger backed observability
to see the recorded scopes, attributes and metrics.

```python
from draive import setup_logging
from haiway import LoggerObservability

setup_logging("basics")  # setup logger

async with ctx.scope(  # prepare the context and see the execution metrics report
    "basics",
    OpenAIResponsesConfig(  # define model configuration for OpenAI Responses API
        model="gpt-5.5",
        reasoning="none",
    ),
    disposables=(OpenAI(),),
    observability=LoggerObservability(),
):
    await text_completion(
        text="Roses are red...",
    )

    with ctx.updating(
        ctx.state(OpenAIResponsesConfig).updating(
            model="gpt-5.6-terra",
        ),
    ):
        await text_completion(
            text="Roses are red...",
        )
```

```text
07/Mar/2025:13:40:51 +0000 [DEBUG] [basics] [8d198c3d552b48f1b7473f1e14ba50ed] [basics] [057c9032-a45a-48f3-8dcb-b7861e4b172e] Entering scope: basics

07/Mar/2025:13:40:52 +0000 [DEBUG] [basics] [8d198c3d552b48f1b7473f1e14ba50ed] [basics] [057c9032-a45a-48f3-8dcb-b7861e4b172e] Exiting scope: basics
07/Mar/2025:13:40:52 +0000 [DEBUG] [basics] [8d198c3d552b48f1b7473f1e14ba50ed] [basics] [057c9032-a45a-48f3-8dcb-b7861e4b172e] Metric - scope_time:1.383s

07/Mar/2025:13:40:52 +0000 [DEBUG] [basics] Observability summary:
┍━ basics [057c9032-a45a-48f3-8dcb-b7861e4b172e]:
┝ Metric - scope_time:1.383s
|  ┍━ text_generation [872ecd2b-612e-4fcb-b19f-4d6aa674e22d]:
|  ┝ Metric - scope_time:0.628s
|  |  ┍━ step.completion.loop [db5e56df-44d2-478c-a1fc-ac959c29bdd3]:
|  |  ┝ Metric - scope_time:0.627s
|  |  |  ┍━ step.completion.loop.iteration_0 [1cbd5f0e-85d5-4b88-a887-56b9f514a0eb]:
|  |  |  ┝ Metric - scope_time:0.626s
|  |  |  |  ┍━ model.invocation [136e0480-cf09-42f7-8450-d0feb8364310]:
|  |  |  |  ┝ Attributes: {
|  |  |  |  |    ["model.provider"]: "openai"
|  |  |  |  |    ["model.name"]: "gpt-5.5"
|  |  |  |  |    ["model.temperature"]: None
|  |  |  |  |    ["model.tools"]:   []
|  |  |  |  |    ["model.tools.selection"]: "none"
|  |  |  |  |    ["model.output"]: "text"
|  |  |  |  |    ["model.stop_sequences"]: None
|  |  |  |  |    ["model.reasoning"]: "none"
|  |  |  |  |    ["model.service_tier"]: "auto"
|  |  |  |  |    ["model.truncation"]: "auto"
|  |  |  |  |  }
|  |  |  |  ┝ Metric: model.input_tokens = 24 tokens
|  |  |  |  |  {
|  |  |  |  |    ["model.provider"]: "openai"
|  |  |  |  |    ["model.name"]: "gpt-5.5"
|  |  |  |  |  }
|  |  |  |  ┝ Metric: model.input_tokens.cached = 0 tokens
|  |  |  |  |  {
|  |  |  |  |    ["model.provider"]: "openai"
|  |  |  |  |    ["model.name"]: "gpt-5.5"
|  |  |  |  |  }
|  |  |  |  ┝ Metric: model.output_tokens = 7 tokens
|  |  |  |  |  {
|  |  |  |  |    ["model.provider"]: "openai"
|  |  |  |  |    ["model.name"]: "gpt-5.5"
|  |  |  |  |  }
|  |  |  |  ┝ Metric - scope_time:0.625s
|  |  |  |  ┕━
|  |  |  ┕━
|  |  ┕━
|  ┕━
|  ┍━ text_generation [38c8e1bd-88a4-4446-81f1-bd2e61bce296]:
|  ┝ Metric - scope_time:0.755s
|  |  ┍━ step.completion.loop [f63a2825-efa6-406f-8019-635aa9892b2e]:
|  |  ┝ Metric - scope_time:0.754s
|  |  |  ┍━ step.completion.loop.iteration_0 [75d4e5de-3a68-4a6d-8f0c-d19c3cf63420]:
|  |  |  ┝ Metric - scope_time:0.753s
|  |  |  |  ┍━ model.invocation [56f8d4f0-3c12-421b-8877-3bbda703cb36]:
|  |  |  |  ┝ Attributes: {
|  |  |  |  |    ["model.provider"]: "openai"
|  |  |  |  |    ["model.name"]: "gpt-5.6-terra"
|  |  |  |  |    ["model.temperature"]: None
|  |  |  |  |    ["model.tools"]:   []
|  |  |  |  |    ["model.tools.selection"]: "none"
|  |  |  |  |    ["model.output"]: "text"
|  |  |  |  |    ["model.stop_sequences"]: None
|  |  |  |  |    ["model.reasoning"]: "none"
|  |  |  |  |    ["model.service_tier"]: "auto"
|  |  |  |  |    ["model.truncation"]: "auto"
|  |  |  |  |  }
|  |  |  |  ┝ Metric: model.input_tokens = 24 tokens
|  |  |  |  |  {
|  |  |  |  |    ["model.provider"]: "openai"
|  |  |  |  |    ["model.name"]: "gpt-5.6-terra"
|  |  |  |  |  }
|  |  |  |  ┝ Metric: model.input_tokens.cached = 0 tokens
|  |  |  |  |  {
|  |  |  |  |    ["model.provider"]: "openai"
|  |  |  |  |    ["model.name"]: "gpt-5.6-terra"
|  |  |  |  |  }
|  |  |  |  ┝ Metric: model.output_tokens = 7 tokens
|  |  |  |  |  {
|  |  |  |  |    ["model.provider"]: "openai"
|  |  |  |  |    ["model.name"]: "gpt-5.6-terra"
|  |  |  |  |  }
|  |  |  |  ┝ Metric - scope_time:0.752s
|  |  |  |  ┕━
|  |  |  ┕━
|  |  ┕━
|  ┕━
┕━
```

The more advanced usage and use cases can be explored in other notebooks.
