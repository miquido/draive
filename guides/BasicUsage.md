# Basic LLM usage

Draive framework provides various ways to use LLM depending on the use case. The simplest interface is to generate text by using `TextGeneration.generate` method. We can use it to make a simple text completion function.


```python
from draive import TextGeneration


async def text_completion(text: str) -> str:
    # TextGeneration.generate is a simple interface for generating text
    return await TextGeneration.generate(
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
    disposables=(OpenAI(),),  # set currently used LMM to OpenAI
    OpenAIChatConfig(model="gpt-4o-mini"), # select used model
):
    result: str = await text_completion(
        text="Roses are red...",
    )

    print(result)
```

    Violets are blue,
    Sugar is sweet,
    And so are you.


As we now know how to setup OpenAI as out LLM provider. We can start customizing it more by providing GPT model configuration.


```python
from draive.openai import OpenAIChatConfig

async with ctx.scope(  # prepare the new context
    "basics",
    disposables=(OpenAI(),),
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
    # when using TextGeneration.generate method directly
    # here we are using gpt-3.5-turbo with temperature of 0.7
    result = await TextGeneration.generate(
        instruction="Prepare simplest completion of given text",
        input="Roses are red...",
        temperature=0.7,
    )

    print("RESULT GPT 3.5 | temperature 0.7:", result)
```

    RESULT GPT 3.5 | temperature 0.4: Violets are blue.


    RESULT GPT 4o | temperature 0.4: Violets are blue.


    RESULT GPT 3.5 | temperature 0.7: Violets are blue.


Since we know the basics, now we can examine the details of our execution to see what actually happened inside. We can setup the logger before execution and assign a logging metrics handler to see context metrics logs.


```python
from draive import setup_logging
from haiway import LoggerObservability

setup_logging("basics")  # setup logger

async with ctx.scope(  # prepare the context and see the execution metrics report
    "basics",
    disposables=(OpenAI(),),
    OpenAIChatConfig(  # define GPT model configuration
        model="gpt-3.5-turbo",
        temperature=0.4,
    ),
    observability=LoggerObservability(),
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

    07/Mar/2025:13:40:51 +0000 [DEBUG] [basics] [8d198c3d552b48f1b7473f1e14ba50ed] [basics] [057c9032a45a48f38dcbb7861e4b172e] Entering context...


    07/Mar/2025:13:40:52 +0000 [DEBUG] [basics] [8d198c3d552b48f1b7473f1e14ba50ed] [basics] [057c9032a45a48f38dcbb7861e4b172e] Metrics summary:
    ⎡ @basics [057c9032a45a48f38dcbb7861e4b172e](1.38s):
    |
    |  ⎡ @generate_text [872ecd2b612e4fcbb19f4d6aa674e22d](0.63s):
    |  |
    |  |  ⎡ @openai_lmm_invocation [db5e56df44d2478ca1fcac959c29bdd3](0.63s):
    |  |  |  ⎡ •ArgumentsTrace:
    |  |  |  |  ├ kwargs:
    |  |  |  |  |  [instruction]: "Prepare the simplest completion of a given text"
    |  |  |  |  |  [context]:
    |  |  |  |  |  |  [0] content:
    |  |  |  |  |  |  |    parts:
    |  |  |  |  |  |  |      - text: Roses are red...
    |  |  |  |  |  |  |        meta: None
    |  |  |  |  |  |  |  meta: None
    |  |  |  |  |  [tool_selection]: "auto"
    |  |  |  |  |  [output]: "text"
    |  |  |  ⌊
    |  |  |  ⎡ •OpenAIChatConfig:
    |  |  |  |  ├ model: "gpt-3.5-turbo"
    |  |  |  |  ├ temperature: 0.4
    |  |  |  ⌊
    |  |  |  ⎡ •TokenUsage:
    |  |  |  |  ├ usage:
    |  |  |  |  |  [gpt-3.5-turbo-0125]:
    |  |  |  |  |  ├ input_tokens: 24
    |  |  |  |  |  ├ cached_tokens: 0
    |  |  |  |  |  ├ output_tokens: 7
    |  |  |  ⌊
    |  |  |  ⎡ •ResultTrace:
    |  |  |  |  ├ result: content:
    |  |  |  |  |    parts:
    |  |  |  |  |      - text: Violets are blue.
    |  |  |  |  |        meta: None
    |  |  |  |  |  meta: None
    |  |  |  ⌊
    |  |  ⌊
    |  ⌊
    |
    |  ⎡ @generate_text [38c8e1bd88a4444681f1bd2e61bce296](0.76s):
    |  |
    |  |  ⎡ @openai_lmm_invocation [f63a2825efa6406f8019635aa9892b2e](0.76s):
    |  |  |  ⎡ •ArgumentsTrace:
    |  |  |  |  ├ kwargs:
    |  |  |  |  |  [instruction]: "Prepare the simplest completion of a given text"
    |  |  |  |  |  [context]:
    |  |  |  |  |  |  [0] content:
    |  |  |  |  |  |  |    parts:
    |  |  |  |  |  |  |      - text: Roses are red...
    |  |  |  |  |  |  |        meta: None
    |  |  |  |  |  |  |  meta: None
    |  |  |  |  |  [tool_selection]: "auto"
    |  |  |  |  |  [output]: "text"
    |  |  |  ⌊
    |  |  |  ⎡ •OpenAIChatConfig:
    |  |  |  |  ├ model: "gpt-4o"
    |  |  |  |  ├ temperature: 0.4
    |  |  |  ⌊
    |  |  |  ⎡ •TokenUsage:
    |  |  |  |  ├ usage:
    |  |  |  |  |  [gpt-4o-2024-08-06]:
    |  |  |  |  |  ├ input_tokens: 24
    |  |  |  |  |  ├ cached_tokens: 0
    |  |  |  |  |  ├ output_tokens: 7
    |  |  |  ⌊
    |  |  |  ⎡ •OpenAISystemFingerprint:
    |  |  |  |  ├ system_fingerprint: "fp_eb9dce56a8"
    |  |  |  ⌊
    |  |  |  ⎡ •ResultTrace:
    |  |  |  |  ├ result: content:
    |  |  |  |  |    parts:
    |  |  |  |  |      - text: Violets are blue.
    |  |  |  |  |        meta: None
    |  |  |  |  |  meta: None
    |  |  |  ⌊
    |  |  ⌊
    |  ⌊
    ⌊


    07/Mar/2025:13:40:52 +0000 [DEBUG] [basics] [8d198c3d552b48f1b7473f1e14ba50ed] [basics] [057c9032a45a48f38dcbb7861e4b172e] ...exiting context after 1.38s


The more advanced usage and use cases can be explored in other notebooks.
