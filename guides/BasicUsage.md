# Basic LLM usage

Draive framework provides various ways to use LLM depending on the use case. The simplest interface is to generate text by using `generate_text` function. We can use it to make a simple text completion function.

```python
from draive import generate_text


async def text_completion(text: str) -> str:
    # generate_text is a simple interface for generating text
    return await generate_text(
        # We have to provide instructions / system prompt to instruct the model
        instruction="Prepare the simplest completion of a given text",
        # input is provided separately
        input=text,
    )
```

The result of this function is a completion from a currently used model. What is a currently used model? We have to define it yet by providing basic setup of state and dependencies. In this example we are going to use OpenAI client, you have to provide the api key to that service in .env file with `OPENAI_API_KEY` key before running.

```python
from draive import load_env

load_env()  # load .env variables
```

When we have .env loaded we can prepare a context scope with OpenAI client and use our function. The lowest-level interface is called LMM. Draive supports multi-model solutions out of the box. Using the `openai_lmm` result as the current scope state selects this provider for our completions.

```python
from draive import ctx
from draive.openai import OpenAIChatConfig, OpenAI

async with ctx.scope(  # prepare new context
    "basics",
    OpenAI().lmm_invoking(),  # set currently used LMM to OpenAI
    OpenAIChatConfig(model="gpt-4o-mini"), # select used model
):
    result: str = await text_completion(
        text="Roses are red...",
    )

    print(result)
```

As we now know how to setup OpenAI as out LLM provider. We can start customizing it more by providing GPT model configuration.

```python
from draive.openai import OpenAIChatConfig, OpenAI

async with ctx.scope(  # prepare the new context
    "basics",
    OpenAI().lmm_invoking(),
    # define GPT model configuration as a context scope state
    OpenAIChatConfig(
        model="gpt-3.5-turbo",
        temperature=0.4,
    ),
):
    # now we are using gpt-3.5-turbo with temperature of 0.4
    result: str = await text_completion(
        text="Roses are red...",
    )

    print("RESULT GPT 3.5 | temperature 0.4:", result)

    # we can update the configuration to change any parameter for nested context
    with ctx.updated(
        # we are updating the current context value instead of making a new one
        # this allows to preserve other elements of the configuration
        ctx.state(OpenAIChatConfig).updated(
            model="gpt-4o",
        ),
    ):
        # now we are using gpt-4o with temperature of 0.4
        result = await text_completion(
            text="Roses are red...",
        )

        print("RESULT GPT 4o | temperature 0.4:", result)

    # we can also update the configuration for a single call
    # when using generate_text function directly
    # here we are using gpt-3.5-turbo with temperature of 0.7
    result = await generate_text(
        instruction="Prepare simplest completion of given text",
        input="Roses are red...",
        temperature=0.7,
    )

    print("RESULT GPT 3.5 | temperature 0.7:", result)
```

Since we know the basics, now we can examine the details of our execution to see what actually happened inside. We can setup the logger before execution and assign a logging metrics handler to see context metrics logs.

```python
from draive import MetricsLogger, setup_logging
from draive.openai import OpenAI
setup_logging("basics")  # setup logger

async with ctx.scope(  # prepare the context and see the execution metrics report
    "basics",
    OpenAI().lmm_invoking(),
    OpenAIChatConfig(  # define GPT model configuration
        model="gpt-3.5-turbo",
        temperature=0.4,
    ),
    metrics=MetricsLogger.handler()
):
    await text_completion(
        text="Roses are red...",
    )

    with ctx.updated(
        ctx.state(OpenAIChatConfig).updated(
            model="gpt-4o",
        ),
    ):
        await text_completion(
            text="Roses are red...",
        )
```

The more advanced usage and use cases can be explored in other notebooks.