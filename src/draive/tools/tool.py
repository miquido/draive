import inspect
import types
import typing
from collections.abc import Callable
from typing import (
    Any,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    final,
    get_args,
    get_origin,
    overload,
)
from uuid import uuid4

from draive.scope import ArgumentsTrace, ResultTrace, ctx
from draive.tools.state import ToolCallContext, ToolsProgressContext
from draive.types import (
    MISSING,
    MissingValue,
    Model,
    ProgressUpdate,
    StringConvertible,
    ToolCallProgress,
    ToolException,
    ToolSpecification,
    parameter_specification,
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
class ToolArgument:
    def __init__(  # noqa: PLR0913
        self,
        name: str,
        alias: str | None,
        description: str | None,
        annotation: Any,
        default: Any | MissingValue,
        validator: Callable[[Any], None] | None,
    ) -> None:
        self.name: str = name
        self.alias: str | None = alias
        self.description: str | None = description
        self.annotation: Any = annotation
        self.default: Any | MissingValue = default
        self.required: bool = default is MISSING
        self.validated: Callable[[Any], Any] = _prepare_validator(
            annotation=annotation,
            additional=validator,
        )


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
        self._arguments: dict[str, ToolArgument] = _arguments(function)
        self._specification: ToolSpecification = {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {
                        argument.alias or argument.name: parameter_specification(
                            annotation=argument.annotation,
                            origin=get_origin(argument.annotation),
                            description=argument.description,
                        )
                        for argument in self._arguments.values()
                    },
                    "required": [
                        argument.alias or argument.name for argument in self._arguments.values()
                    ],
                },
                "description": description or "",
            },
        }

    @property
    def name(self) -> str:
        return self._name

    @property
    def specification(self) -> ToolSpecification:
        return self._specification

    @property
    def available(self) -> bool:
        return self._availability()

    def validated(
        self,
        *,
        arguments: dict[str, Any],
        strict: bool = False,
    ) -> dict[str, Any]:
        for argument in self._arguments.values():
            if value := arguments.get(argument.name):
                arguments[argument.name] = argument.validated(value)
            elif (alias := argument.alias) and (value := arguments.get(alias)):
                del arguments[alias]  # remove aliased value
                arguments[argument.name] = argument.validated(value)
            elif argument.default is not MISSING:
                arguments[argument.name] = argument.default
            else:
                raise ValueError("Missing required argument", argument.name)
        unexpected: set[str] = set(self._arguments.keys()).difference(arguments.keys())
        if strict and unexpected:
            raise ValueError("Unexpected arguments provided", unexpected)
        else:
            for key in unexpected:
                del arguments[key]
        return arguments

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
        async with ctx.nested(
            self._name,
            ArgumentsTrace(call_id=call_context.call_id, **kwargs),
        ):
            with ctx.updated(call_context):
                progress(
                    ToolCallProgress(
                        call_id=call_context.call_id,
                        tool=call_context.tool,
                        status="STARTED",
                        content=None,
                    )
                )
                if not self.available:
                    progress(
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
                    arguments = self.validated(arguments=kwargs)
                except (ValueError, TypeError) as exc:
                    progress(
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
                await ctx.record(ResultTrace(result))
                progress(
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


_ParameterType_T = TypeVar("_ParameterType_T")


def Parameter(  # Ruff - noqa: B008
    *,
    alias: str | None = None,
    description: str | None = None,
    default: _ParameterType_T | MissingValue = MISSING,
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
        default: Any | MissingValue,
        validator: Callable[[Any], None] | None,
    ) -> None:
        self.alias: str | None = alias
        self.description: str | None = description
        self.default: Any | MissingValue = default
        self.validator: Callable[[Any], None] | None = validator


def _prepare_validator(  # noqa: C901
    annotation: Any,
    additional: Callable[[Any], None] | None,
) -> Callable[[Any], Any]:
    match get_origin(annotation) or annotation:
        case typing.Annotated:
            match get_args(annotation):
                case [annotated, *_]:
                    return _prepare_validator(
                        annotation=annotated,
                        additional=additional,
                    )
                case annotated:
                    raise TypeError("Unsupported annotated type", annotated)

        case typing.Literal:

            def validated(value: Any) -> Any:
                if value in get_args(annotation):
                    if validate := additional:
                        validate(value)
                    return value
                else:
                    raise TypeError("Invalid value", annotation, value)

            return validated

        case types.UnionType | typing.Union:
            validators: list[Callable[[Any], Any]] = [
                _prepare_validator(
                    annotation=alternative,
                    additional=additional,
                )
                for alternative in get_args(annotation)
            ]

            def validated(value: Any) -> Any:
                for validator in validators:
                    try:
                        return validator(value)
                    except TypeError:
                        continue  # check next alternative

                raise TypeError("Invalid value", annotation, value)

            return validated

        case model_type if issubclass(model_type, Model):

            def validated(value: Any) -> Any:
                model: Model
                if isinstance(value, dict):
                    model = model_type.from_dict(value=cast(dict[str, Any], value))
                elif isinstance(value, model_type):
                    model = value
                else:
                    raise TypeError("Invalid value", annotation, value)
                if validate := additional:
                    validate(model)
                return model

            return validated

        case typed_dict_type if typing.is_typeddict(typed_dict_type):

            def validated(value: Any) -> Any:
                typed_dict: dict[Any, Any]
                if isinstance(value, dict):
                    typed_dict = typed_dict_type(**value)
                else:
                    raise TypeError("Invalid value", annotation, value)
                if validate := additional:
                    validate(typed_dict)
                return typed_dict

            return validated

        case other_type:

            def validated(value: Any) -> Any:
                if isinstance(value, other_type):
                    if validate := additional:
                        validate(value)
                    return value
                elif isinstance(value, float) and other_type == int:
                    # auto convert float to int - json does not distinguish those
                    converted_int: int = int(value)
                    if validate := additional:
                        validate(converted_int)
                    return converted_int
                elif isinstance(value, int) and other_type == float:
                    # auto convert int to float - json does not distinguish those
                    converted_float: float = float(value)
                    if validate := additional:
                        validate(converted_float)
                    return converted_float
                # TODO: validate function/callable values
                elif callable(value):
                    if validate := additional:
                        validate(value)
                    return value
                else:
                    raise TypeError("Invalid value", annotation, value)

            return validated


def _arguments(
    function: ToolFunction[ToolArgs, ToolResult_co],
    /,
) -> dict[str, ToolArgument]:
    arguments: dict[str, ToolArgument] = {}

    for parameter in inspect.signature(function).parameters.values():
        if parameter.annotation is inspect._empty:  # pyright: ignore[reportPrivateUsage]
            # skip object method "self" argument
            if parameter.name != "self":
                raise TypeError(
                    "Untyped argument %s",
                    parameter.name,
                )
        elif isinstance(parameter.default, ToolParameter):
            arguments[parameter.name] = ToolArgument(
                name=parameter.name,
                alias=parameter.default.alias,
                description=parameter.default.description,
                default=parameter.default.default,
                annotation=parameter.annotation,
                validator=parameter.default.validator,
            )
        else:
            arguments[parameter.name] = ToolArgument(
                name=parameter.name,
                alias=None,
                description=None,
                default=MISSING
                if parameter.default is inspect._empty  # pyright: ignore[reportPrivateUsage]
                else parameter.default,
                annotation=parameter.annotation,
                validator=None,
            )

    return arguments
