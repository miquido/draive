from collections.abc import Callable, Coroutine, Mapping
from typing import Any, Protocol, cast, final, overload

from haiway import ArgumentsTrace, ResultTrace, ctx

from draive.commons import META_EMPTY, Meta
from draive.lmm.types import (
    LMMException,
    LMMToolError,
    LMMToolFunctionSpecification,
    LMMToolSpecification,
)
from draive.multimodal import Multimodal, MultimodalContent, MultimodalContentConvertible
from draive.parameters import ParameterSpecification, ParametersSpecification, ParametrizedFunction

__all__ = [
    "AnyTool",
    "Tool",
    "ToolAvailabilityCheck",
    "tool",
]


class ToolAvailabilityCheck(Protocol):
    def __call__(self) -> bool: ...


@final
class Tool[**Args, Result](ParametrizedFunction[Args, Coroutine[None, None, Result]]):
    __slots__ = (
        "_check_availability",
        "_direct_result",
        "description",
        "format_failure",
        "format_result",
        "meta",
        "name",
        "specification",
    )

    def __init__(  # noqa: PLR0913
        self,
        /,
        name: str,
        *,
        function: Callable[Args, Coroutine[None, None, Result]],
        description: str | None,
        specification: ParametersSpecification | None,
        availability_check: ToolAvailabilityCheck | None,
        format_result: Callable[[Result], Multimodal],
        format_failure: Callable[[Exception], Multimodal],
        direct_result: bool = False,
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

        self.specification: LMMToolSpecification = LMMToolFunctionSpecification(
            name=name,
            description=description,
            parameters=specification,
        )

        self.name: str = name
        self.description: str | None = description
        self._direct_result: bool = direct_result
        self._check_availability: ToolAvailabilityCheck = availability_check or (
            lambda: True  # available by default
        )
        self.format_result: Callable[[Result], Multimodal] = format_result
        self.format_failure: Callable[[Exception], Multimodal] = format_failure
        self.meta: Meta = meta

    @property
    def available(self) -> bool:
        try:
            return self._check_availability()

        except Exception:
            return False

    @property
    def requires_direct_result(self) -> bool:
        return self._direct_result

    # call from toolbox
    async def _toolbox_call(
        self,
        call_id: str,
        /,
        arguments: Mapping[str, Any],
    ) -> MultimodalContent:
        with ctx.scope(self.name):
            ctx.record(ArgumentsTrace.of(**arguments))
            try:
                try:
                    if not self.available:
                        raise LMMException(f"Tool {self.name} is not available!")

                    result: Result = await super().__call__(**arguments)  # pyright: ignore[reportCallIssue]
                    ctx.record(ResultTrace.of(result))

                    return MultimodalContent.of(self.format_result(result))

                except Exception as exc:
                    # return an error with formatted content
                    raise LMMToolError(
                        f"Tool {self.name}[{call_id}] failed",
                        content=MultimodalContent.of(self.format_failure(exc)),
                    ) from exc

            except BaseException as exc:
                ctx.record(ResultTrace.of(exc))
                ctx.log_error(
                    "Tool call error",
                    exception=exc,
                )
                raise exc

    # regular call when using as a function
    async def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        with ctx.scope(self.name):
            ctx.record(ArgumentsTrace.of(*args, **kwargs))
            try:
                result: Result = await super().__call__(
                    *args,
                    **kwargs,
                )
                ctx.record(ResultTrace.of(result))

                return result

            except BaseException as exc:
                ctx.record(ResultTrace.of(exc))
                ctx.log_error(
                    "Tool call error",
                    exception=exc,
                )
                raise exc


AnyTool = Tool[Any, Any]


class ToolWrapper(Protocol):
    def __call__[**Args, Result](
        self,
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> Tool[Args, Result]: ...


class PartialToolWrapper[Result](Protocol):
    def __call__[**Args](
        self,
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> Tool[Args, Result]: ...


@overload
def tool[**Args, Result](
    function: Callable[Args, Coroutine[None, None, Result]],
    /,
) -> Tool[Args, Result]:
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
    availability_check: ToolAvailabilityCheck | None = None,
    format_result: Callable[[Result], Multimodal],
    format_failure: Callable[[Exception], Multimodal] | None = None,
    direct_result: bool = False,
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
    availability_check: ToolAvailabilityCheck
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
    direct_result: bool
        controls if tool result should break the ongoing processing and be the direct result of it.
        Note that during concurrent execution of multiple tools the call/result order defines
        direct result and exact behavior is not defined.
        Default is False.
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
    availability_check: ToolAvailabilityCheck | None = None,
    format_failure: Callable[[Exception], Multimodal] | None = None,
    direct_result: bool = False,
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
    availability_check: ToolAvailabilityCheck
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
    direct_result: bool
        controls if tool result should break the ongoing processing and be the direct result of it.
        Note that during concurrent execution of multiple tools the call/result order defines
        direct result and exact behavior is not defined.
        Default is False.
    meta: Meta | None
        custom metadata allowing to access tool metadata like its source in case of remote tools.

    Returns
    -------
    Callable[[Function[Args, Coroutine[None, None, Result]]], Tool[Args, Result]]
        function allowing to convert other function to a Tool using provided configuration.
    """


def tool[**Args, Result](  # noqa: PLR0913
    function: Callable[Args, Coroutine[None, None, Result]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability_check: ToolAvailabilityCheck | None = None,
    format_result: Callable[[Result], Multimodal] | None = None,
    format_failure: Callable[[Exception], Multimodal] | None = None,
    direct_result: bool = False,
    meta: Meta | None = None,
) -> PartialToolWrapper[Result] | ToolWrapper | Tool[Args, Result]:
    def wrap[**Arg](
        function: Callable[Arg, Coroutine[None, None, Result]],
    ) -> Tool[Arg, Result]:
        return Tool[Arg, Result](
            name=name or function.__name__,
            description=description,
            specification=None,
            function=function,
            availability_check=availability_check,
            format_result=format_result or _default_result_format,
            format_failure=format_failure or _default_failure_result,
            direct_result=direct_result,
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


def _default_failure_result(exception: Exception) -> MultimodalContent:
    return MultimodalContent.of("ERROR")
