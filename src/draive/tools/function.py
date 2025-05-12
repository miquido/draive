from collections.abc import Callable, Coroutine
from typing import Any, Protocol, cast, final, overload

from haiway import ctx
from haiway.utils import format_str

from draive.commons import META_EMPTY, Meta
from draive.lmm.types import (
    LMMToolError,
)
from draive.multimodal import MultimodalContent, MultimodalContentConvertible
from draive.parameters import ParameterSpecification, ParametersSpecification, ParametrizedFunction
from draive.tools.types import (
    ToolAvailabilityChecking,
    ToolErrorFormatting,
    ToolHandling,
    ToolResultFormatting,
)

__all__ = ("tool",)


@final
class FunctionTool[**Args, Result](ParametrizedFunction[Args, Coroutine[None, None, Result]]):
    __slots__ = (
        "_check_availability",
        "_format_failure",
        "_format_result",
        "description",
        "handling",
        "meta",
        "name",
        "parameters",
    )

    def __init__(
        self,
        /,
        name: str,
        *,
        function: Callable[Args, Coroutine[None, None, Result]],
        description: str | None,
        specification: ParametersSpecification | None,
        availability_check: ToolAvailabilityChecking | None,
        format_result: ToolResultFormatting[Result],
        format_failure: ToolErrorFormatting,
        handling: ToolHandling = "auto",
        meta: Meta,
    ) -> None:
        super().__init__(function)

        if specification is None:
            aliased_required: list[str] = []
            parameters: dict[str, ParameterSpecification] = {}
            for parameter in self._parameters.values():
                parameters[parameter.alias or parameter.name] = parameter.specification

                if parameter.required:
                    aliased_required.append(parameter.alias or parameter.name)

            specification = {
                "type": "object",
                "properties": parameters,
                "required": aliased_required,
            }

        if not specification["properties"]:
            specification = None  # use no parameters without arguments

        self.name: str = name
        self.description: str | None = description
        self.parameters: ParametersSpecification | None = specification
        self.handling: ToolHandling = handling
        self._check_availability: ToolAvailabilityChecking = availability_check or (
            lambda: True  # available by default
        )
        self._format_result: ToolResultFormatting[Result] = format_result
        self._format_failure: ToolErrorFormatting = format_failure
        self.meta: Meta = meta

    @property
    def available(self) -> bool:
        try:
            return self._check_availability()

        except Exception:
            return False

    # call as a tool
    async def tool_call(
        self,
        call_id: str,
        /,
        **arguments: Any,
    ) -> MultimodalContent:
        with ctx.scope(self.name):
            ctx.record(
                attributes={
                    "call_id": call_id,
                    **{key: f"{arg}" for key, arg in arguments.items()},
                }
            )
            try:
                try:
                    result: Result = await super().__call__(**arguments)  # pyright: ignore[reportCallIssue]
                    formatted_result: MultimodalContent = MultimodalContent.of(
                        self._format_result(result)
                    )

                    ctx.record(
                        event="result",
                        attributes={"value": format_str(formatted_result)},
                    )

                    return formatted_result

                except Exception as exc:
                    # return an error with formatted content
                    raise LMMToolError(
                        f"Tool {self.name}[{call_id}] call failed due to an error:"
                        f" {type(exc)} {exc}",
                        content=MultimodalContent.of(self._format_failure(exc)),
                    ) from exc

            except BaseException as exc:
                ctx.record(
                    event="result",
                    attributes={"error": f"{type(exc)}: {exc}"},
                )
                ctx.log_error(
                    "Tool call exception",
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
    availability_check: ToolAvailabilityChecking | None = None,
    format_result: ToolResultFormatting[Result],
    format_failure: ToolErrorFormatting | None = None,
    handling: ToolHandling = "auto",
    meta: Meta | None = None,
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
    availability_check: ToolAvailabilityChecking
        function used to verify availability of the tool in given context. It can be used to check
        permissions or occurrence of a specific state to allow its usage.
        Provided function should raise an Exception when the tool should not be available.
        Default is always available.
    format_result: Callable[[Result], MultimodalContent]
        function converting tool result to MultimodalContent. It is used to format the result
        for model processing. Default implementation converts the result to string if needed.
    format_failure: Callable[[Exception], MultimodalContent]
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
    availability_check: ToolAvailabilityChecking | None = None,
    format_failure: ToolErrorFormatting | None = None,
    handling: ToolHandling = "auto",
    meta: Meta | None = None,
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
    availability_check: ToolAvailabilityChecking
        function used to verify availability of the tool in given context. It can be used to check
        permissions or occurrence of a specific state to allow its usage.
        Provided function should raise an Exception when the tool should not be available.
        Default is always available.
    format_failure: Callable[[Exception], MultimodalContent]
        function converting tool call exception to a fallback MultimodalContent.
        Default implementation return "ERROR" string and logs the exception.
    handling: ToolHandling
        controls if tool result should break the ongoing processing and be the direct result of it.
        Note that during concurrent execution of multiple tools the call/result order defines
        direct result and exact behavior is not defined.
        Default is "auto".
    meta: Meta | None
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
    availability_check: ToolAvailabilityChecking | None = None,
    format_result: ToolResultFormatting[Result] | None = None,
    format_failure: ToolErrorFormatting | None = None,
    handling: ToolHandling = "auto",
    meta: Meta | None = None,
) -> PartialToolWrapper[Result] | ToolWrapper | FunctionTool[Args, Result]:
    def wrap[**Arg](
        function: Callable[Arg, Coroutine[None, None, Result]],
    ) -> FunctionTool[Arg, Result]:
        return FunctionTool[Arg, Result](
            name=name or function.__name__,
            description=description,
            specification=None,
            function=function,
            availability_check=availability_check,
            format_result=format_result or _default_result_format,
            format_failure=format_failure or _default_failure_result,
            handling=handling,
            meta=meta if meta is not None else META_EMPTY,
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


def _default_failure_result(error: Exception) -> MultimodalContent:
    return MultimodalContent.of("ERROR")
