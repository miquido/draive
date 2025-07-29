# Basics of draive

Draive is a framework for building high-quality applications utilizing LLMs. It provides a set of tools allowing to easily manage data flows, dependencies, and state propagation across the application. Draive is suitable for building servers, workers, and local solutions including CLI and GUI based. The core principles of the framework are strongly connected to functional programming and structured concurrency concepts. All of the draive code is fully typed as strictly as possible making as much use of the static analysis as possible. This means that numerous potential issues can be caught before running the code. It is strongly recommended to use strict, full-type linting if possible. Before you dive deeply into the code you should familiarize yourself with the basics.

## Defining state

Let's start by defining a state that can be used within the application. The state can be an LLM configuration, details of the current user, etc. We have two base classes available in draive: `State` and `DataModel` both have a similar basic behavior. The difference is that the `DataModel` is meant to be serializable and have an associated schema description, while the `State` does not. This distinction allows us to propagate non serializable data i.e. functions when using the `State`. Let's define a simple state:


```python
from draive import State


# inherit from the base class
class BasicState(State):
    # fields are automatically converted to instance properties
    identifier: str
    value: int
```

Both `State` and `DataModel` utilize transformation similar to Python's dataclass or pydantic BaseModel which means that all properties defined for the class will be converted into the instance properties and an appropriate `__init__` function will be generated as well. Additionally, both types come with built-in validation. Each object is automatically validated during the initialization ensuring proper types and values for each field. Both types are also immutable by default - you will receive linting and runtime errors when trying to mutate those. To make a mutation of an instance of either of those we can use a dedicated `updated` method which makes a copy on the fly and validates mutation as well. Let's have a look:


```python
# create an instance of BasicState
basic_state: BasicState = BasicState(
    identifier="basic",
    value=42,
)

# BasicState(
#     identifier=0, # error! can't instantiate state with wrong data
#     value=42,
# )

# prepare an update
updated_state: BasicState = basic_state.updated(
    value=21
)  # value of `identifier` field won't change

# basic_state.value = 0 # error! can't mutate the state
```

When it comes to the `DataModel` type in addition to `State` features, we can use JSON serialization to directly create an instance. We can access its associated schema as well. `DataModel` has also more strict validation including nested structures validation and automatic type conversion.


```python
from collections.abc import Sequence

from draive import DataModel


# prepare a class, inherit from DataModel this time
class BasicModel(DataModel):
    username: str
    tags: Sequence[str] | None = None


json: str = """\
{
  "username": "John Doe",
  "tags": ["example", "json"]
}\
"""

# note that the model will be fully validated during decoding
decoded_model: BasicModel = BasicModel.from_json(json)
print(f"Decoded model:\n{decoded_model}")

# we can also get the json schema of the model
print(f"JSON Schema:\n{BasicModel.json_schema(indent=2)}")
```

    Decoded model:
    username: John Doe
    tags:
      - example
      - json
    JSON Schema:
    {
      "type": "object",
      "properties": {
        "username": {
          "type": "string"
        },
        "tags": {
          "oneOf": [
            {
              "type": "array",
              "items": {
                "type": "string"
              }
            },
            {
              "type": "null"
            }
          ]
        }
      },
      "required": [
        "username"
      ]
    }


For more details about `State` and `DataModel` please see the [AdvancedState](./AdvancedState.md)

## Propagating state
We have defined our application state, now it is time to learn how to propagate it through the application. Draive comes with the `ctx` helper for managing the contextual state and dependencies propagation. Most of the draive functions require to be called inside a context scope. Each meaningful operation execution should be wrapped into a separate, new context. Let's define one now:


```python
from draive import ctx

# ctx.scope creates a new instance of the context scope
# we can enter it by using async context manager:
async with ctx.scope("basics"):
    pass # now everything executed inside will use that scope
```

Ok, we have defined our first context scope, but it is not useful yet. Since we already defined a state, let's see how to propagate it. When defining a function we typically define its arguments to access the state within its scope:


```python
def do_something(state: BasicState) -> None:
    pass # use the state here
```

It is still the go-to solution for accessing the data in a function scope. However, when a piece of data has to propagate through various function calls, it can be annoying to add an extra argument everywhere, especially in the functions that do not require it by itself. A common solution would be to introduce a shared global state to which all functions have access to, but this has some significant drawbacks. First, it can't be mutated locally to have a different value only for a subset of functions. Otherwise, we introduce a globally shared, mutable state, which is a common source of bugs. When using draive you can propagate any state through the context scope and change it locally when needed. We can begin by defining the initial state:


```python
# when defining context scope we can provide any number of state objects,
# there can be only a single instance for any given type though
async with ctx.scope("basics", basic_state):
    pass  # now we are using the defined state in this scope

# out of the scope, state is not defined here
```

Each state type can have a single instance available in the context. Its value can be accessed by using the type as a key with proper Generic (parametrized types) handling. Note that only subclasses of `State` can be used as context state, `DataModel` has a different purpose and cannot be used that way. Now we can see how to access it inside the function:


```python
def do_something_contextually() -> None:
    # access contextual state through the ctx
    state: BasicState = ctx.state(BasicState)
    print(state) # then you can use it in the scope of the function
```

What will be the current state is defined by the context of the execution. We can change it locally by using another function from `ctx` called `updated` which allows us to update the state by copying the context and allowing to enter a new scope with it:


```python
async with ctx.scope("basics", basic_state):
    # here we have access to the state from the context
    print("Initial:")
    do_something_contextually()

    # then we can update it locally
    with ctx.updated(basic_state.updated(identifier="updated")):
        print("Updated:")
        # and access its updated version
        do_something_contextually()

    print("Final:")
    # when leaving the updated scope we go back to previously defined state
    do_something_contextually()

# do_something_contextually() # calling it outside of any context scope will cause an error
```

    Initial:
    BasicState(identifier: basic, value: 42)
    Updated:
    BasicState(identifier: updated, value: 42)
    Final:
    BasicState(identifier: basic, value: 42)


## Logs and metrics

Each context scope comes with an additional hidden feature - metrics. Each new context scope creates a new associated metrics scope as well. Metrics allow gathering of diagnostic information about program execution. Each scope gains an unique identifier and can have an additional label that allows to easily find associated information in the logs. Speaking about the logs - those are covered by the context scope as well! You can use `ctx` to log information with additional metadata included. Finally, you can define custom metrics and nested stages, and define how all of the metrics will be consumed. Let's have a look:


```python
# we can assign label to the context to better identify it
async with ctx.scope("logging"):
    ctx.log_info("We can use associated logger with additional metadata")
    # finally we can record custom metrics based on the state we already know
    # by using a subclass of `State`
    ctx.record(BasicState(identifier="recorded", value=42))
```

Nice, but we can't see anything here. How about configuring the loggers first? Draive comes with a helper function for that:


```python
from draive import setup_logging

# setup python loggers and add a new logger with a given name
setup_logging("logging")
```

Since we are into the helpers for a bit - here is one more useful helper for loading the .env files:


```python
from draive import load_env

load_env()
```

Those two helpers should be called when initializing your application to ensure proper environment and logging. Now let's go back to the metrics. When dealing with metrics, we have to define an observability handler; otherwise, all metrics will be simply logged. Draive comes with a predefined logger observability called `LoggerObservability` converting all metrics into logs with a call tree summary at the scope exit:


```python
from haiway import LoggerObservability

async with ctx.scope(
    # we can explicitly pass the logger as an argument
    # or use the assigned label as the default logger name
    "logging",
    # define the scope observability to log the scope details
    observability=LoggerObservability(),
):
    ctx.log_info("Now we can see the logs!")
    # we can see recorded custom metrics for the context scope as well
    ctx.record(BasicState(identifier="recorded", value=42))

    # or we can create nested context scope which creates
    # a separate subtree for metrics
    with ctx.scope("nested"):
        # now we can record the same thing again
        # but this time it will be associated with the nested context scope
        ctx.record(BasicState(identifier="recorded-nested", value=11))

# additionally when we exit the context scope we get the execution summary
# including all recorded metrics
```

    07/Mar/2025:13:40:54 +0000 [DEBUG] [logging] [73243ab2691a4753a85a4ddf23643b47] [logging] [488ece9d384e4b6fac258d87ca78d52c] Entering context...


    07/Mar/2025:13:40:54 +0000 [INFO] [logging] [73243ab2691a4753a85a4ddf23643b47] [logging] [488ece9d384e4b6fac258d87ca78d52c] Now we can see the logs!


    07/Mar/2025:13:40:54 +0000 [DEBUG] [logging] [73243ab2691a4753a85a4ddf23643b47] [logging] [488ece9d384e4b6fac258d87ca78d52c] Recorded metric:
    ⎡ BasicState:
    ├ identifier: "recorded"
    ├ value: 42
    ⌊


    07/Mar/2025:13:40:54 +0000 [DEBUG] [nested] [73243ab2691a4753a85a4ddf23643b47] [nested] [aa01c8e4dad44122bfa79447d964f016] Entering context...


    07/Mar/2025:13:40:54 +0000 [DEBUG] [nested] [73243ab2691a4753a85a4ddf23643b47] [nested] [aa01c8e4dad44122bfa79447d964f016] Recorded metric:
    ⎡ BasicState:
    ├ identifier: "recorded-nested"
    ├ value: 11
    ⌊


    07/Mar/2025:13:40:54 +0000 [DEBUG] [nested] [73243ab2691a4753a85a4ddf23643b47] [nested] [aa01c8e4dad44122bfa79447d964f016] ...exiting context after 0.00s


    07/Mar/2025:13:40:54 +0000 [DEBUG] [logging] [73243ab2691a4753a85a4ddf23643b47] [logging] [488ece9d384e4b6fac258d87ca78d52c] Metrics summary:
    ⎡ @logging [488ece9d384e4b6fac258d87ca78d52c](0.00s):
    |  ⎡ •BasicState:
    |  |  ├ identifier: "recorded"
    |  |  ├ value: 42
    |  ⌊
    |
    |  ⎡ @nested [aa01c8e4dad44122bfa79447d964f016](0.00s):
    |  |  ⎡ •BasicState:
    |  |  |  ├ identifier: "recorded-nested"
    |  |  |  ├ value: 11
    |  |  ⌊
    |  ⌊
    ⌊


    07/Mar/2025:13:40:54 +0000 [DEBUG] [logging] [73243ab2691a4753a85a4ddf23643b47] [logging] [488ece9d384e4b6fac258d87ca78d52c] ...exiting context after 0.00s


Now that is useful! You can customize how those metrics are reported and gathered by implementing the custom MetricsHandler and using it when creating a new scope context. This allows you to use any custom metrics-gathering tool you like.

We have covered all basic concepts related to draive framework. Familiarizing with those is very important and allows us to build on top of that foundation to provide elegant and simple solutions for complex problems. This is not the end of our journey though, there are a lot more details, helpers, and customizations to explore!
