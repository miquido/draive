from collections.abc import Callable
from typing import Generic, ParamSpec, Protocol, TypeVar, final, overload

from draive.helpers import extract_specification
from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.tools.errors import ToolException
from draive.types import StringConvertible, ToolSpecification

__all__ = [
    "Tool",
    "ToolAvailability",
    "tool",
    "redefine_tool",
    "ToolArgs",
    "ToolResult_co",
    "ToolFunction",
]


ToolArgs = ParamSpec(
    name="ToolArgs",
    # bound= - ideally it should be bound to allowed types, not implemented in python yet
)


ToolResult_co = TypeVar(
    name="ToolResult_co",
    bound=StringConvertible,
    covariant=True,
)


class ToolFunction(Protocol[ToolArgs, ToolResult_co]):
    @property
    def __name__(self) -> str:
        ...

    async def __call__(
        self,
        *args: ToolArgs.args,
        **kwargs: ToolArgs.kwargs,
    ) -> ToolResult_co:
        ...


class ToolAvailability(Protocol):
    def __call__(self) -> bool:
        ...


@final
class Tool(Generic[ToolArgs, ToolResult_co]):
    def __init__(
        self,
        /,
        name: str,
        *,
        function: ToolFunction[ToolArgs, ToolResult_co],
        description: str | None = None,
        availability: ToolAvailability | None = None,
    ):
        assert not isinstance(  # nosec: B101
            function, Tool
        ), "Tools should not be used as functions for tools"
        self._name: str = name
        self._description: str | None = description
        self._availability: ToolAvailability = availability or (
            lambda: True  # available by default
        )
        self._function_call: ToolFunction[ToolArgs, ToolResult_co] = function
        self._specification: ToolSpecification = {
            "type": "function",
            "function": {
                "name": name,
                "parameters": extract_specification(function),
            },
        }
        if description:
            self._specification["function"]["description"] = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def specification(self) -> ToolSpecification:
        return self._specification

    @property
    def available(self) -> bool:
        return self._availability()

    async def __call__(
        self,
        *args: ToolArgs.args,
        **kwargs: ToolArgs.kwargs,
    ) -> ToolResult_co:
        assert not args, "Positional unkeyed arguments are not supported"  # nosec: B101

        async with ctx.nested(
            self._name,
            ArgumentsTrace(**kwargs),
        ):
            if not self.available:
                raise ToolException("Attempting to use unavailable tool")

            result: ToolResult_co = await self._function_call(
                *args,
                **kwargs,
            )

            await ctx.record(ResultTrace(result))

            return result


@overload
def tool(
    function: ToolFunction[ToolArgs, ToolResult_co],
    /,
) -> Tool[ToolArgs, ToolResult_co]:
    ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailability | None = None,
) -> Callable[[ToolFunction[ToolArgs, ToolResult_co]], Tool[ToolArgs, ToolResult_co]]:
    ...


def tool(
    function: ToolFunction[ToolArgs, ToolResult_co] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailability | None = None,
) -> (
    Callable[[ToolFunction[ToolArgs, ToolResult_co]], Tool[ToolArgs, ToolResult_co]]
    | Tool[ToolArgs, ToolResult_co]
):
    def wrap(function: ToolFunction[ToolArgs, ToolResult_co]) -> Tool[ToolArgs, ToolResult_co]:
        return Tool(
            name=name or function.__name__,
            description=description,
            function=function,
            availability=availability,
        )

    if function := function:
        return wrap(
            function=function,
        )
    else:
        return wrap


def redefine_tool(
    tool: Tool[ToolArgs, ToolResult_co],
    /,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailability | None = None,
) -> Tool[ToolArgs, ToolResult_co]:
    return Tool(
        name=name or tool.name,
        function=tool._function_call,  # pyright: ignore[reportPrivateUsage]
        description=description or tool._description,  # pyright: ignore[reportPrivateUsage]
        availability=availability or tool._availability,  # pyright: ignore[reportPrivateUsage]
    )
