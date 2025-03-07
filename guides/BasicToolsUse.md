# Basic tools use

Various LLM use cases can utilize function/tool calls to extend LLM capabilities. Draive comes with a dedicated solution to prepare tools and control its execution.

## Tool definition

Let's start by defining a simple tool. Tool is usually a function which is explained to LLM and can be requested to be used. Tools can require generation of arguments and have to return some value. Tools defined within draive are python async function annotated with `tool` wrapper.

```python
from draive import tool


@tool # simply annotate function as a tool, tools can have arguments using basic types
async def current_time(location: str) -> str:
    # return result of tool execution, we are using fake time here
    return f"Time in {location} is 9:53:22"
```

## Tool use

After defining a tool we can use it in multiple scenarios to extend LLM capabilities. All of the tool arguments will be validated when calling. It includes type check and any additional validation if defined. You can still use it as a regular function, although, all tools have to be executed within draive context scope. 

```python
from draive import ctx

async with ctx.scope("basics"):
    # we can still use it as a regular function
    # but it has to be executed within context scope
    print(await current_time(location="London"))

# await current_time(location="London") # error! out of context
```

The biggest benefit of defining a tool is when using LLM. We can tell the model to use available tools to extend its capabilities. Higher level interfaces automatically handle tool calls and going back with its result to LLM to receive the final result. We can see how it works within a simple text generator. Tools are provided within an iterable collection like list or tuple or within the `Toolbox` object which allows customizing tools execution. You can prepare tools collection each time it is used or reuse preconstructed one. We will use OpenAI GPT model as it natively supports tool use. Make sure to provide .env file with `OPENAI_API_KEY` key before running. 

```python
from draive import generate_text, load_env
from draive.openai import OpenAIChatConfig, OpenAI

load_env()

async with ctx.scope(
    "basics",
    # define used LMM to be OpenAI within the context
    OpenAI().lmm_invoking(),
    OpenAIChatConfig(model="gpt-4o-mini"),
):
    result: str = await generate_text(
        instruction="You are a helpful assistant",
        input="What is the time in New York?",
        tools=[current_time], # or `Toolbox(current_time)`
    )

    print(result)
```

## Tool details

Tools can be customized and extended in various ways depending on use case. First of all we can customize tool arguments and help LLM to better understand how to use given tool.

```python
from draive import Argument


@tool( # this time we will use additional arguments within tool annotation
    # we can define an alias used as the tool name when explaining it to LLM,
    # default name is the name of python function
    name="fun_fact",
    # additionally we can explain the tool purpose by using description
    description="Find a fun fact in a given topic",
)
async def customized(
    # we can also annotate arguments to provide even more details
    # and specify argument handling logic
    arg: str = Argument(
        # we can alias each argument name
        aliased="topic",
        # further describe its use
        description="Topic of a fact to find",
        # provide default value or default value factory
        default="random",
        # end more, including custom validators
    ),
) -> str:
    return f"{arg} is very funny on its own!"

# we can examine tool specification which is similar to
# how `State` and `DataModel` specification/schema is built
print(customized.specification)
```

We can customize tools even more using additional parameters like custom result formatting or requiring direct result from tool when used.

```python
@tool( # this time we will use different arguments within tool annotation
    # direct result allows returning the result without additional LLM call
    direct_result=True,
    # format_result allows altering the way LLMs will see tool results
    # we can also use format_failure to format tool issues
    format_result=lambda result: f"Formatted: {result}",
    # we can also contextually limit tool availability in runtime
    # by removing it from available tools list on given condition
    availability_check=lambda: True
)
async def customized_more() -> str:
    return "to be changed"
```

## Toolbox

We have already mentioned the Toolbox which allows us to specify some additional details regarding the tools execution like the tool calls limit or a tool suggestion. Here is how it looks like:

```python
from draive import Toolbox
from draive.openai import OpenAI

async with ctx.scope(
    "basics",
    # define used LMM to be OpenAI within the context
    OpenAI().lmm_invoking(),
    OpenAIChatConfig(model="gpt-4o-mini"),
):
    result: str = await generate_text(
        instruction="You are a funny assistant",
        input="What is the funny thing about LLMs?",
        tools=Toolbox.of(
            # we can define any number of tools within a toolbox
            current_time,
            # we can also force any given tool use within the first LLM call
            suggest=customized,
            # we can limit how many tool calls are allowed
            # before the final result is returned
            repeated_calls_limit=2,
        ),
    )

    print(result)
```

## Metrics

All of the tool usage is automatically traced within scope metrics. We can see the details about their execution when using a logger:

```python
from draive import MetricsLogger, setup_logging
from draive.openai import OpenAI

setup_logging("basics")

async with ctx.scope(
    "basics",
    # define used LMM to be OpenAI within the context
    OpenAI().lmm_invoking(),
    OpenAIChatConfig(model="gpt-4o-mini"),
    metrics=MetricsLogger.handler(),
):
    result: str = await generate_text(
        instruction="You are a funny assistant",
        input="What is the funny thing about LLMs?",
        # we will now be able to see what tools were used
        # and check the details about its execution
        tools=Toolbox.of(
            current_time,
            suggest=customized,
        ),
    )

    print(f"\nResult:\n{result}\n")
```