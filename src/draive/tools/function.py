from collections.abc import Callable, Coroutine
from typing import Any, Protocol, Self, cast, final, overload

from haiway import Meta, MetaValues, ctx
from haiway.utils import format_str

from draive.lmm import LMMToolSpecification
from draive.multimodal import MultimodalContent, MultimodalContentConvertible
from draive.parameters import ParameterSpecification, ParametersSpecification, ParametrizedFunction
from draive.tools.types import (
    ToolAvailabilityChecking,
    ToolError,
    ToolErrorFormatting,
    ToolHandling,
    ToolResultFormatting,
)

__all__ = ("tool",)


@final
class FunctionTool[**Args, Result](ParametrizedFunction[Args, Coroutine[None, None, Result]]):
    __slots__ = (
        "_check_availability",
        "description",
        "format_error",
        "format_result",
        "handling",
        "meta",
        "name",
        "parameters",
        "specification",
    )

    def __init__(
        self,
        /,
        function: Callable[Args, Coroutine[None, None, Result]],
        *,
        name: str,
        description: str | None,
        parameters: ParametersSpecification | None,
        availability: ToolAvailabilityChecking | None,
        format_result: ToolResultFormatting[Result],
        format_error: ToolErrorFormatting,
        handling: ToolHandling = "auto",
        meta: Meta,
    ) -> None:
        super().__init__(function)

        if parameters is None:
            aliased_required: list[str] = []
            specifications: dict[str, ParameterSpecification] = {}
            for parameter in self._parameters.values():
                specifications[parameter.alias or parameter.name] = parameter.specification

                if parameter.required:
                    aliased_required.append(parameter.alias or parameter.name)

            parameters = {
                "type": "object",
                "properties": specifications,
                "required": aliased_required,
            }

        if not parameters["properties"]:
            parameters = None  # use no parameters without arguments

        self.name: str
        object.__setattr__(
            self,
            "name",
            name,
        )
        self.description: str | None
        object.__setattr__(
            self,
            "description",
            description,
        )
        self.parameters: ParametersSpecification | None
        object.__setattr__(
            self,
            "parameters",
            parameters,
        )
        self.specification: LMMToolSpecification
        object.__setattr__(
            self,
            "specification",
            {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        )
        self.handling: ToolHandling
        object.__setattr__(
            self,
            "handling",
            handling,
        )
        self._check_availability: ToolAvailabilityChecking
        object.__setattr__(
            self,
            "_check_availability",
            availability or _available,  # available by default
        )
        self.format_result: ToolResultFormatting[Result]
        object.__setattr__(
            self,
            "format_result",
            format_result,
        )
        self.format_error: ToolErrorFormatting
        object.__setattr__(
            self,
            "format_error",
            format_error,
        )
        self.meta: Meta
        if description:
            object.__setattr__(
                self,
                "meta",
                meta.updated(
                    kind="tool",
                    name=name,
                    description=description,
                ),
            )

        else:
            object.__setattr__(
                self,
                "meta",
                meta.updated(
                    kind="tool",
                    name=name,
                ),
            )

    def updated(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        function: Callable[Args, Coroutine[None, None, Result]] | None = None,
        parameters: ParametersSpecification | None = None,
        availability: ToolAvailabilityChecking | None = None,
        format_result: ToolResultFormatting[Result] | None = None,
        format_error: ToolErrorFormatting | None = None,
        handling: ToolHandling | None = None,
        meta: Meta | None = None,
    ) -> Self:
        return self.__class__(
            function or self._call,
            name=name or self.name,
            description=description or self.description,
            parameters=parameters or self.parameters,
            availability=availability or self._check_availability,
            format_result=format_result or self.format_result,
            format_error=format_error or self.format_error,
            handling=handling or self.handling,
            meta=meta or self.meta,
        )

    def available(
        self,
        tools_turn: int,
    ) -> bool:
        try:
            return self._check_availability(
                tools_turn=tools_turn,
                meta=self.meta,
            )

        except Exception as e:
            ctx.log_error(
                f"Availability check of tool ({self.name}) failed, tool will be unavailable.",
                exception=e,
            )
            return False

    # call as a tool
    async def call(
        self,
        call_id: str,
        /,
        **arguments: Any,
    ) -> MultimodalContent:
        async with ctx.scope(f"tool.{self.name}"):
            ctx.record(
                attributes={
                    "call_id": call_id,
                    **{key: f"{arg}" for key, arg in arguments.items()},
                }
            )

            try:
                result: Result = await super().__call__(**arguments)  # pyright: ignore[reportCallIssue]
                formatted_result: MultimodalContent = MultimodalContent.of(
                    self.format_result(result)
                )

                ctx.record(
                    event="result",
                    attributes={"value": format_str(formatted_result)},
                )

                return formatted_result

            except Exception as exc:
                ctx.record(
                    event="error",
                    attributes={
                        "exception": f"{type(exc)}",
                        "message": f"{exc}",
                    },
                )
                error_message: str = f"Tool {self.name} call [{call_id}] failed due to an error"
                ctx.log_error(
                    error_message,
                    exception=exc,
                )
                # raise an error with formatted content
                raise ToolError(
                    error_message,
                    content=MultimodalContent.of(self.format_error(exc)),
                ) from exc

            # do not catch and format BaseExceptions, rethrow instead
            except BaseException as exc:
                ctx.record(
                    event="error",
                    attributes={
                        "exception": f"{type(exc)}",
                        "message": f"{exc}",
                    },
                )
                ctx.log_error(
                    f"Tool {self.name} call [{call_id}] failed due to an error",
                    exception=exc,
                )
                raise exc

    # regular call when using as a function
    async def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        return await super().__call__(
            *args,
            **kwargs,
        )


def _available(
    tools_turn: int,
    meta: Meta,
) -> bool:
    return True


class ToolWrapper(Protocol):
    def __call__[**Args, Result](
        self,
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> FunctionTool[Args, Result]: ...


class PartialToolWrapper[Result](Protocol):
    def __call__[**Args](
        self,
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> FunctionTool[Args, Result]: ...


@overload
def tool[**Args, Result](
    function: Callable[Args, Coroutine[None, None, Result]],
    /,
) -> FunctionTool[Args, Result]:
    """
    Convert a function to a tool using default parameters and no description.

    In order to adjust the arguments behavior and specification use an instance of Argument
    as a default value of any given argument with desired configuration
    for each argument individually.

    Parameters
    ----------
    function: Callable[Args, Coroutine[None, None, Result]]
        a function to be wrapped as a Tool.
    Returns
    -------
    Tool[Args, Result]
        a Tool representation of the provided function.
    """


@overload
def tool[Result](
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailabilityChecking | None = None,
    format_result: ToolResultFormatting[Result],
    format_error: ToolErrorFormatting | None = None,
    handling: ToolHandling = "auto",
    meta: Meta | MetaValues | None = None,
) -> PartialToolWrapper[Result]:
    """
    Convert a function to a tool using provided parameters.

    In order to adjust the arguments behavior and specification use an instance of Argument
    as a default value of any given argument with desired configuration
    for each argument individually.

    Parameters
    ----------
    name: str
        name to be used in a tool specification.
        Default is the name of the wrapped function.
    description: int
        description to be used in a tool specification. Allows to present the tool behavior to the
        external system.
        Default is empty.
    availability: ToolAvailabilityChecking
        function used to verify availability of the tool in given context. It can be used to check
        permissions or occurrence of a specific state to allow its usage.
        Provided function should raise an Exception when the tool should not be available.
        Default is always available.
    format_result: Callable[[Result], MultimodalContent]
        function converting tool result to MultimodalContent. It is used to format the result
        for model processing. Default implementation converts the result to string if needed.
    format_error: Callable[[Exception], MultimodalContent]
        function converting tool call exception to a fallback MultimodalContent.
        Default implementation return "ERROR" string and logs the exception.
    handling: ToolHandling
        controls if tool result should break the ongoing processing and be the direct result of it.
        Note that during concurrent execution of multiple tools the call/result order defines
        direct result and exact behavior is not defined.
        Default is "auto".
    meta: Mapping[str, str | float | int | bool | None] | None
        custom metadata allowing to access tool metadata like its source in case of remote tools.

    Returns
    -------
    Callable[[Function[Args, Coroutine[None, None, Result]]], Tool[Args, Result]]
        function allowing to convert other function to a Tool using provided configuration.
    """


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailabilityChecking | None = None,
    format_error: ToolErrorFormatting | None = None,
    handling: ToolHandling = "auto",
    meta: Meta | MetaValues | None = None,
) -> ToolWrapper:
    """
    Convert a function to a tool using provided parameters.

    In order to adjust the arguments behavior and specification use an instance of Argument
    as a default value of any given argument with desired configuration
    for each argument individually.

    Parameters
    ----------
    name: str
        name to be used in a tool specification.
        Default is the name of the wrapped function.
    description: int
        description to be used in a tool specification. Allows to present the tool behavior to the
        external system.
        Default is empty.
    availability: ToolAvailabilityChecking
        function used to verify availability of the tool in given context. It can be used to check
        permissions or occurrence of a specific state to allow its usage.
        Provided function should raise an Exception when the tool should not be available.
        Default is always available.
    format_error: Callable[[Exception], MultimodalContent]
        function converting tool call exception to a fallback MultimodalContent.
        Default implementation return "ERROR" string and logs the exception.
    handling: ToolHandling
        controls if tool result should break the ongoing processing and be the direct result of it.
        Note that during concurrent execution of multiple tools the call/result order defines
        direct result and exact behavior is not defined.
        Default is "auto".
    meta: Meta | MetaValues | None
        custom metadata allowing to access tool metadata like its source in case of remote tools.

    Returns
    -------
    Callable[[Function[Args, Coroutine[None, None, Result]]], Tool[Args, Result]]
        function allowing to convert other function to a Tool using provided configuration.
    """


def tool[**Args, Result](
    function: Callable[Args, Coroutine[None, None, Result]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailabilityChecking | None = None,
    format_result: ToolResultFormatting[Result] | None = None,
    format_error: ToolErrorFormatting | None = None,
    handling: ToolHandling = "auto",
    meta: Meta | MetaValues | None = None,
) -> PartialToolWrapper[Result] | ToolWrapper | FunctionTool[Args, Result]:
    def wrap[**Arg](
        function: Callable[Arg, Coroutine[None, None, Result]],
    ) -> FunctionTool[Arg, Result]:
        return FunctionTool[Arg, Result](
            function=function,
            name=name or function.__name__,
            description=description,
            parameters=None,
            availability=availability,
            format_result=format_result or _default_result_format,
            format_error=format_error or _default_error_format,
            handling=handling,
            meta=Meta.of(meta),
        )

    if function := function:
        return wrap(function=function)

    else:
        return cast(PartialToolWrapper[Result], wrap)


def _default_result_format(result: Any) -> MultimodalContent:
    match result:
        case MultimodalContent() as content:
            return content

        case element if isinstance(element, MultimodalContentConvertible):
            return MultimodalContent.of(element)

        case [*elements]:
            return MultimodalContent.of(
                *[
                    element if isinstance(element, MultimodalContentConvertible) else str(element)
                    for element in elements
                ]
            )

        case other:
            return MultimodalContent.of(str(other))


def _default_error_format(error: Exception) -> MultimodalContent:
    return MultimodalContent.of("ERROR")
