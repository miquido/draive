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

After defining a tool, we can use it in multiple scenarios to extend LLM capabilities. All tool arguments are validated when calling. It includes type check and any additional validation if defined. You can still use it as a regular function, although, all tools have to be executed within draive context scope.


```python
from draive import ctx

async with ctx.scope("basics"):
    # we can still use it as a regular function
    # but it has to be executed within context scope
    print(await current_time(location="London"))

# await current_time(location="London") # error! out of context
```

    Time in London is 9:53:22


The biggest benefit of defining a tool becomes apparent when using an LLM. We can instruct the model to use available tools to extend its capabilities. Higher-level interfaces automatically handle tool calls and return results to the LLM to generate the final output.
We can observe how this works within a simple text generator. Tools are provided through an iterable collection such as a list or tuple, or within a Toolbox object that allows for customized tool execution. You can prepare a tools collection each time it is used, or you can reuse a preconstructed one.
We will use the OpenAI GPT model since it natively supports tool use. Make sure to provide an .env file with an OPENAI_API_KEY key before running the code.


```python
from draive import TextGeneration, load_env
from draive.openai import OpenAIChatConfig, OpenAI

load_env()

async with ctx.scope(
    "basics",
    # define used LMM to be OpenAI within the context
    OpenAIChatConfig(model="gpt-4o-mini"),
    disposables=(OpenAI(),),
):
    result: str = await TextGeneration.generate(
        instruction="You are a helpful assistant",
        input="What is the time in New York?",
        tools=[current_time], # or `Toolbox(current_time)`
    )

    print(result)
```

    The current time in New York is 9:53 AM.


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

    {'name': 'fun_fact', 'description': 'Find a fun fact in a given topic', 'parameters': {'type': 'object', 'properties': {'topic': {'type': 'string', 'description': 'Topic of a fact to find'}}, 'required': []}}


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

async with ctx.scope(
    "basics",
    # define used LMM to be OpenAI within the context
    OpenAIChatConfig(model="gpt-4o-mini"),
    disposables=(OpenAI(),),
):
    result: str = await TextGeneration.generate(
        instruction="You are a funny assistant",
        input="What is the funny thing about LLMs?",
        tools=Toolbox.of(
            # we can define any number of tools within a toolbox
            current_time,
            customized,
            # we can also force any given tool use within the first LLM call
            suggest=customized,
            # we can limit how many tool calls are allowed
            # before the final result is returned
            repeated_calls_limit=2,
        ),
    )

    print(result)
```

    The funny thing about large language models (LLMs) is that they can generate text that sounds incredibly smart, yet they often confuse "there," "their," and "they're" like it's a game! It's like having a genius friend who also thinks a duck is a variety of potato. Who knew AI could be so amusing?


## Metrics

All of the tool usage is automatically traced within scope metrics. We can see the details about their execution when using a logger:


```python
from haiway import LoggerObservability
from draive import setup_logging

setup_logging("basics")

async with ctx.scope(
    "basics",
    # define used LMM to be OpenAI within the context
    OpenAIChatConfig(model="gpt-4o-mini"),
    disposables=(OpenAI(),),
    observability=LoggerObservability(),
):
    result: str = await TextGeneration.generate(
        instruction="You are a funny assistant",
        input="What is the funny thing about LLMs?",
        # we will now be able to see what tools were used
        # and check the details about its execution
        tools=Toolbox.of(
            current_time,
            customized,
            suggest=customized,
        ),
    )

    print(f"\nResult:\n{result}\n")
```

    07/Mar/2025:13:40:43 +0000 [DEBUG] [basics] [8ffed39264134e679e87ae5a34311ad8] [basics] [24eee7a7577b4f25b7c59a43c20a3423] Entering context...



    Result:
    One funny thing about LLMs (Large Language Models) is that they can generate text that sounds like it was written by a human, but sometimes they mix up facts in the most amusing ways! For example, they might describe a cat as "a small, furry creature that loves to swim" – which is definitely not the case for most cats! They have a knack for making up their own versions of reality, which can lead to some hilarious misunderstandings!

    07/Mar/2025:13:40:47 +0000 [DEBUG] [basics] [8ffed39264134e679e87ae5a34311ad8] [basics] [24eee7a7577b4f25b7c59a43c20a3423] Metrics summary:
    ⎡ @basics [24eee7a7577b4f25b7c59a43c20a3423](3.82s):
    |
    |  ⎡ @generate_text [484bad5216bf4a2db28dad924f5d0de9](3.82s):
    |  |
    |  |  ⎡ @openai_lmm_invocation [aea1a47734504cc38267ea31a42aa31e](1.04s):
    |  |  |  ⎡ •ArgumentsTrace:
    |  |  |  |  ├ kwargs:
    |  |  |  |  |  [instruction]: "You are a funny assistant"
    |  |  |  |  |  [context]:
    |  |  |  |  |  |  [0] content:
    |  |  |  |  |  |  |    parts:
    |  |  |  |  |  |  |      - text: What is the funny thing about LLMs?
    |  |  |  |  |  |  |        meta: None
    |  |  |  |  |  |  |  meta: None
    |  |  |  |  |  |  [1] completion: None
    |  |  |  |  |  |  |  requests:
    |  |  |  |  |  |  |    - identifier: call_QpvQQ89llJxvuuW7rmMY9CqA
    |  |  |  |  |  |  |      tool: fun_fact
    |  |  |  |  |  |  |      arguments:
    |  |  |  |  |  |  |        topic: LLMs
    |  |  |  |  |  |  [2] responses:
    |  |  |  |  |  |  |    - identifier: call_QpvQQ89llJxvuuW7rmMY9CqA
    |  |  |  |  |  |  |      tool: fun_fact
    |  |  |  |  |  |  |      content:
    |  |  |  |  |  |  |        parts:
    |  |  |  |  |  |  |          - text: LLMs is very funny on its own!
    |  |  |  |  |  |  |            meta: None
    |  |  |  |  |  |  |      direct: False
    |  |  |  |  |  |  |      error: False
    |  |  |  |  |  [tool_selection]:
    |  |  |  |  |  |  [name]: "fun_fact"
    |  |  |  |  |  |  [description]: "Find a fun fact in a given topic"
    |  |  |  |  |  |  [parameters]:
    |  |  |  |  |  |  |  [type]: "object"
    |  |  |  |  |  |  |  [properties]:
    |  |  |  |  |  |  |  |  [topic]:
    |  |  |  |  |  |  |  |  |  [type]: "string"
    |  |  |  |  |  |  |  |  |  [description]: "Topic of a fact to find"
    |  |  |  |  |  [tools]:
    |  |  |  |  |  |  [0]
    |  |  |  |  |  |  |  [name]: "current_time"
    |  |  |  |  |  |  |  [description]: None
    |  |  |  |  |  |  |  [parameters]:
    |  |  |  |  |  |  |  |  [type]: "object"
    |  |  |  |  |  |  |  |  [properties]:
    |  |  |  |  |  |  |  |  |  [location]:
    |  |  |  |  |  |  |  |  |  |  [type]: "string"
    |  |  |  |  |  |  |  |  [required]:
    |  |  |  |  |  |  |  |  |  [0] "location"
    |  |  |  |  |  |  [1]
    |  |  |  |  |  |  |  [name]: "fun_fact"
    |  |  |  |  |  |  |  [description]: "Find a fun fact in a given topic"
    |  |  |  |  |  |  |  [parameters]:
    |  |  |  |  |  |  |  |  [type]: "object"
    |  |  |  |  |  |  |  |  [properties]:
    |  |  |  |  |  |  |  |  |  [topic]:
    |  |  |  |  |  |  |  |  |  |  [type]: "string"
    |  |  |  |  |  |  |  |  |  |  [description]: "Topic of a fact to find"
    |  |  |  |  |  [output]: "text"
    |  |  |  ⌊
    |  |  |  ⎡ •OpenAIChatConfig:
    |  |  |  |  ├ model: "gpt-4o-mini"
    |  |  |  |  ├ temperature: 1.0
    |  |  |  ⌊
    |  |  |  ⎡ •TokenUsage:
    |  |  |  |  ├ usage:
    |  |  |  |  |  [gpt-4o-mini-2024-07-18]:
    |  |  |  |  |  ├ input_tokens: 93
    |  |  |  |  |  ├ cached_tokens: 0
    |  |  |  |  |  ├ output_tokens: 7
    |  |  |  ⌊
    |  |  |  ⎡ •OpenAISystemFingerprint:
    |  |  |  |  ├ system_fingerprint: "fp_06737a9306"
    |  |  |  ⌊
    |  |  |  ⎡ •ResultTrace:
    |  |  |  |  ├ result: completion: None
    |  |  |  |  |  requests:
    |  |  |  |  |    - identifier: call_QpvQQ89llJxvuuW7rmMY9CqA
    |  |  |  |  |      tool: fun_fact
    |  |  |  |  |      arguments:
    |  |  |  |  |        topic: LLMs
    |  |  |  ⌊
    |  |  ⌊
    |  |
    |  |  ⎡ @fun_fact [1344c3efabdb4f9a8387c164662f1770](0.00s):
    |  |  |  ⎡ •ArgumentsTrace:
    |  |  |  |  ├ kwargs:
    |  |  |  |  |  [topic]: "LLMs"
    |  |  |  ⌊
    |  |  |  ⎡ •ResultTrace:
    |  |  |  |  ├ result: "LLMs is very funny on its own!"
    |  |  |  ⌊
    |  |  ⌊
    |  |
    |  |  ⎡ @openai_lmm_invocation [1f0d7b3c69aa40bca7eb48fbd36ca4d6](2.78s):
    |  |  |  ⎡ •ArgumentsTrace:
    |  |  |  |  ├ kwargs:
    |  |  |  |  |  [instruction]: "You are a funny assistant"
    |  |  |  |  |  [context]:
    |  |  |  |  |  |  [0] content:
    |  |  |  |  |  |  |    parts:
    |  |  |  |  |  |  |      - text: What is the funny thing about LLMs?
    |  |  |  |  |  |  |        meta: None
    |  |  |  |  |  |  |  meta: None
    |  |  |  |  |  |  [1] completion: None
    |  |  |  |  |  |  |  requests:
    |  |  |  |  |  |  |    - identifier: call_QpvQQ89llJxvuuW7rmMY9CqA
    |  |  |  |  |  |  |      tool: fun_fact
    |  |  |  |  |  |  |      arguments:
    |  |  |  |  |  |  |        topic: LLMs
    |  |  |  |  |  |  [2] responses:
    |  |  |  |  |  |  |    - identifier: call_QpvQQ89llJxvuuW7rmMY9CqA
    |  |  |  |  |  |  |      tool: fun_fact
    |  |  |  |  |  |  |      content:
    |  |  |  |  |  |  |        parts:
    |  |  |  |  |  |  |          - text: LLMs is very funny on its own!
    |  |  |  |  |  |  |            meta: None
    |  |  |  |  |  |  |      direct: False
    |  |  |  |  |  |  |      error: False
    |  |  |  |  |  [tool_selection]: "auto"
    |  |  |  |  |  [tools]:
    |  |  |  |  |  |  [0]
    |  |  |  |  |  |  |  [name]: "current_time"
    |  |  |  |  |  |  |  [description]: None
    |  |  |  |  |  |  |  [parameters]:
    |  |  |  |  |  |  |  |  [type]: "object"
    |  |  |  |  |  |  |  |  [properties]:
    |  |  |  |  |  |  |  |  |  [location]:
    |  |  |  |  |  |  |  |  |  |  [type]: "string"
    |  |  |  |  |  |  |  |  [required]:
    |  |  |  |  |  |  |  |  |  [0] "location"
    |  |  |  |  |  |  [1]
    |  |  |  |  |  |  |  [name]: "fun_fact"
    |  |  |  |  |  |  |  [description]: "Find a fun fact in a given topic"
    |  |  |  |  |  |  |  [parameters]:
    |  |  |  |  |  |  |  |  [type]: "object"
    |  |  |  |  |  |  |  |  [properties]:
    |  |  |  |  |  |  |  |  |  [topic]:
    |  |  |  |  |  |  |  |  |  |  [type]: "string"
    |  |  |  |  |  |  |  |  |  |  [description]: "Topic of a fact to find"
    |  |  |  |  |  [output]: "text"
    |  |  |  ⌊
    |  |  |  ⎡ •OpenAIChatConfig:
    |  |  |  |  ├ model: "gpt-4o-mini"
    |  |  |  |  ├ temperature: 1.0
    |  |  |  ⌊
    |  |  |  ⎡ •TokenUsage:
    |  |  |  |  ├ usage:
    |  |  |  |  |  [gpt-4o-mini-2024-07-18]:
    |  |  |  |  |  ├ input_tokens: 116
    |  |  |  |  |  ├ cached_tokens: 0
    |  |  |  |  |  ├ output_tokens: 95
    |  |  |  ⌊
    |  |  |  ⎡ •OpenAISystemFingerprint:
    |  |  |  |  ├ system_fingerprint: "fp_06737a9306"
    |  |  |  ⌊
    |  |  |  ⎡ •ResultTrace:
    |  |  |  |  ├ result: content:
    |  |  |  |  |    parts:
    |  |  |  |  |      - text: One funny thing about LLMs (Large Language Models) is that they can generate text that sounds like it was written by a human, but sometimes they mix up facts in the most amusing ways! For example, they might describe a cat as "a small, furry creature that loves to swim" – which is definitely not the case for most cats! They have a knack for making up their own versions of reality, which can lead to some hilarious misunderstandings!
    |  |  |  |  |        meta: None
    |  |  |  |  |  meta: None
    |  |  |  ⌊
    |  |  ⌊
    |  ⌊
    ⌊


    07/Mar/2025:13:40:47 +0000 [DEBUG] [basics] [8ffed39264134e679e87ae5a34311ad8] [basics] [24eee7a7577b4f25b7c59a43c20a3423] ...exiting context after 3.82s
