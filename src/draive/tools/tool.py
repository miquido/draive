import inspect
from collections.abc import Callable
from typing import (
    Any,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    final,
    overload,
)
from uuid import uuid4

from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.tools.state import ToolCallContext, ToolsProgressContext
from draive.types import (
    MISSING,
    MissingValue,
    ParameterDefinition,
    ParameterSpecification,
    ProgressUpdate,
    ToolCallProgress,
    ToolException,
    ToolSpecification,
)

__all__ = [
    "Tool",
    "ToolAvailability",
    "tool",
    "ToolArgs",
    "ToolResult_co",
    "ToolFunction",
    "Parameter",
]

ToolArgs = ParamSpec(
    name="ToolArgs",
    # bound= - ideally it should be bound to allowed types, not implemented in python yet
)
ToolResult_co = TypeVar(
    name="ToolResult_co",
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


_ParameterType_T = TypeVar("_ParameterType_T")


def Parameter(  # Ruff - noqa: B008
    *,
    alias: str | None = None,
    description: str | None = None,
    default: Callable[[], _ParameterType_T] | _ParameterType_T | MissingValue = MISSING,
    validator: Callable[[_ParameterType_T], None] | None = None,
) -> _ParameterType_T:  # it is actually a ToolParameter, but type checker has to be fooled
    return cast(
        _ParameterType_T,
        ToolParameter(
            alias=alias,
            description=description,
            default=default,
            validator=validator,
        ),
    )


@final
class ToolParameter:
    def __init__(
        self,
        alias: str | None,
        description: str | None,
        default: Callable[[], _ParameterType_T] | _ParameterType_T | MissingValue,
        validator: Callable[[_ParameterType_T], None] | None,
    ) -> None:
        self.alias: str | None = alias
        self.description: str | None = description
        self.default: Callable[[], _ParameterType_T] | _ParameterType_T | MissingValue = default
        self.validator: Callable[[_ParameterType_T], None] | None = validator


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
        ), "Nested tool wrapping is not allowed"
        # mimic function attributes if able
        try:
            self.__module__ = function.__module__
        except AttributeError:
            pass
        try:
            # python function name may differ from tool name
            self.__name__ = function.__name__
        except AttributeError:
            pass
        try:
            self.__qualname__ = function.__qualname__
        except AttributeError:
            pass
        try:
            self.__doc__ = function.__doc__
        except AttributeError:
            pass
        try:
            self.__annotations__ = function.__annotations__
        except AttributeError:
            pass
        try:
            self.__dict__.update(function.__dict__)
        except AttributeError:
            pass

        self._name: str = name
        self._description: str | None = description
        self._availability: ToolAvailability = availability or (
            lambda: True  # available by default
        )
        self._function_call: ToolFunction[ToolArgs, ToolResult_co] = function
        parameters, parameters_specification, required_names = _function_parameters(function)
        self._parameters: list[ParameterDefinition] = parameters
        self._specification: ToolSpecification = {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": parameters_specification,
                    "required": required_names,
                },
                "description": description or "",
            },
        }

        def frozen(
            __name: str,
            __value: Any,
        ) -> None:
            raise RuntimeError("Tool can't be modified")

        self.__setattr__ = frozen

    @property
    def name(self) -> str:
        return self._name

    @property
    def specification(self) -> ToolSpecification:
        return self._specification

    @property
    def available(self) -> bool:
        return self._availability()

    def _validated(
        self,
        *,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        for parameter in self._parameters:
            if parameter.name in arguments:
                validated[parameter.name] = parameter.validated_value(arguments.get(parameter.name))
            elif (alias := parameter.alias) and (alias in arguments):
                validated[parameter.name] = parameter.validated_value(arguments.get(alias))
            else:
                default_value = parameter.default_value()
                if default_value is MISSING:
                    raise ValueError("Missing required argument", parameter.name)
                else:
                    validated[parameter.name] = default_value

        return validated

    async def __call__(
        self,
        tool_call_id: str | None = None,
        *args: ToolArgs.args,
        **kwargs: ToolArgs.kwargs,
    ) -> ToolResult_co:
        assert not args, "Positional unkeyed arguments are not supported"  # nosec: B101
        call_context: ToolCallContext = ToolCallContext(
            call_id=tool_call_id or uuid4().hex,
            tool=self._name,
        )
        progress: ProgressUpdate[ToolCallProgress] = ctx.state(ToolsProgressContext).progress
        with ctx.nested(
            self._name,
            ArgumentsTrace(call_id=call_context.call_id, **kwargs),
        ):
            with ctx.updated(call_context):
                progress(  # notify on start
                    ToolCallProgress(
                        call_id=call_context.call_id,
                        tool=call_context.tool,
                        status="STARTED",
                        content=None,
                    )
                )
                if not self.available:
                    progress(  # notify on fail
                        ToolCallProgress(
                            call_id=call_context.call_id,
                            tool=call_context.tool,
                            status="FAILED",
                            content=None,
                        )
                    )
                    raise ToolException("Attempting to use unavailable tool")

                arguments: dict[str, Any]
                try:
                    arguments = self._validated(arguments=kwargs)
                except (ValueError, TypeError) as exc:
                    progress(  # notify on fail
                        ToolCallProgress(
                            call_id=call_context.call_id,
                            tool=call_context.tool,
                            status="FAILED",
                            content=None,
                        )
                    )
                    raise ToolException("Tool arguments invalid", self.name) from exc

                result: ToolResult_co = await self._function_call(
                    *args,
                    **arguments,
                )
                ctx.record(ResultTrace(result))
                progress(  # notify on finish
                    ToolCallProgress(
                        call_id=call_context.call_id,
                        tool=call_context.tool,
                        status="FINISHED",
                        content=None,
                    )
                )

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
    """
    Convert a function to a tool. Tool arguments support only limited types.
    """

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


def _function_parameters(
    function: Callable[..., Any],
    /,
) -> tuple[list[ParameterDefinition], dict[str, ParameterSpecification], list[str]]:
    parameters: list[ParameterDefinition] = []
    specification: dict[str, ParameterSpecification] = {}
    required: list[str] = []
    # convert function signature to parameters
    for function_parameter in inspect.signature(function).parameters.values():
        if function_parameter.annotation is inspect._empty:  # pyright: ignore[reportPrivateUsage]
            # skip object method "self" argument
            if function_parameter.name != "self":
                raise TypeError(
                    "Untyped argument %s",
                    function_parameter.name,
                )
            else:
                continue  # skip `self` argument

        parameter: ParameterDefinition
        if isinstance(function_parameter.default, ToolParameter):
            parameter = ParameterDefinition(
                name=function_parameter.name,
                alias=function_parameter.default.alias,
                description=function_parameter.default.description,
                default=function_parameter.default.default,
                annotation=function_parameter.annotation,
                validator=function_parameter.default.validator,
            )
            if function_parameter.default.default is MISSING:
                required.append(parameter.name)
        else:  # use regular annotation
            parameter = ParameterDefinition(
                name=function_parameter.name,
                alias=None,
                description=None,
                default=MISSING
                if function_parameter.default is inspect._empty  # pyright: ignore[reportPrivateUsage]
                else function_parameter.default,
                annotation=function_parameter.annotation,
                validator=None,
            )
            if function_parameter.default is inspect._empty:  # pyright: ignore[reportPrivateUsage]
                required.append(parameter.name)

        assert parameter.name not in specification, f"Parameter duplicate: {parameter.name}"  # nosec: B101
        assert parameter.alias not in specification, f"Parameter alias conflict: {parameter.name}"  # nosec: B101
        parameters.append(parameter)
        specification[parameter.name] = parameter.specification()
    return (parameters, specification, required)
