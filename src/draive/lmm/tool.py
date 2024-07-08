from collections.abc import Callable, Coroutine
from typing import (
    Any,
    Protocol,
    cast,
    final,
    overload,
)
from uuid import uuid4

from draive.lmm.errors import ToolError, ToolException
from draive.lmm.state import ToolCallContext, ToolStatusStream
from draive.metrics import ArgumentsTrace, ResultTrace
from draive.parameters import (
    ParameterSpecification,
    ParametrizedFunction,
    ToolSpecification,
)
from draive.scope import ctx
from draive.types import MultimodalContent, MultimodalContentConvertible
from draive.utils import freeze, noop, not_missing

__all__ = [
    "AnyTool",
    "tool",
    "Tool",
    "ToolAvailabilityCheck",
]


class ToolAvailabilityCheck(Protocol):
    def __call__(self) -> bool: ...


@final
class Tool[**Args, Result](ParametrizedFunction[Args, Coroutine[Any, Any, Result]]):
    def __init__(  # noqa: PLR0913
        self,
        /,
        name: str,
        *,
        function: Callable[Args, Coroutine[Any, Any, Result]],
        description: str | None = None,
        availability_check: ToolAvailabilityCheck | None = None,
        format_result: Callable[[Result], MultimodalContent | MultimodalContentConvertible],
        format_failure: Callable[[Exception], MultimodalContent | MultimodalContentConvertible],
        direct_result: bool = False,
    ) -> None:
        super().__init__(function=function)
        aliased_required: list[str] = []
        parameters: dict[str, ParameterSpecification] = {}
        for parameter in self._parameters.values():
            if not_missing(parameter.specification):
                parameters[parameter.aliased or parameter.name] = parameter.specification

            else:
                raise TypeError(
                    f"{function.__qualname__} can't be represented as a tool"
                    f" - argument '{parameter.name}' is missing specification."
                )

            if not (parameter.has_default or parameter.allows_missing):
                aliased_required.append(parameter.aliased or parameter.name)

        self.specification: ToolSpecification = {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": aliased_required,
                },
                "description": description or "",
            },
        }

        self.name: str = name
        self._direct_result: bool = direct_result
        self._check_availability: ToolAvailabilityCheck = availability_check or (
            lambda: True  # available by default
        )
        self.format_result: Callable[[Result], MultimodalContent | MultimodalContentConvertible] = (
            format_result
        )
        self.format_failure: Callable[
            [Exception], MultimodalContent | MultimodalContentConvertible
        ] = format_failure

        freeze(self)

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
        arguments: dict[str, Any],
    ) -> MultimodalContent:
        call_context: ToolCallContext = ToolCallContext(
            call_id=call_id,
            tool=self.name,
            send_status=ctx.state(ToolStatusStream).send or noop,
        )
        with ctx.nested(
            self.name,
            state=[call_context],
            metrics=[ArgumentsTrace.of(**arguments)],
        ):
            await call_context.report("STARTED")

            try:
                if not self.available:
                    raise ToolException(f"{self.name} is not available!")

                result: Result = await super().__call__(**arguments)  # pyright: ignore[reportCallIssue]
                ctx.record(ResultTrace.of(result))

                await call_context.report("FINISHED")

                return MultimodalContent.of(self.format_result(result))

            except Exception as exc:
                await call_context.report("FAILED")
                # return an error with formatted content
                raise ToolError(
                    f"Tool {self.name}[{call_id}] failed",
                    content=MultimodalContent.of(self.format_failure(exc)),
                ) from exc

    # regular call when using as a function
    async def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        with ctx.nested(
            self.name,
            state=[
                ToolCallContext(
                    call_id=uuid4().hex,
                    tool=self.name,
                    send_status=noop,
                )
            ],
            metrics=[ArgumentsTrace.of(*args, **kwargs)],
        ):
            result: Result = await super().__call__(
                *args,
                **kwargs,
            )

            ctx.record(ResultTrace.of(result))

            return result


AnyTool = Tool[Any, Any]


class ToolWrapper(Protocol):
    def __call__[**Args, Result](
        self,
        function: Callable[Args, Coroutine[Any, Any, Result]],
    ) -> Tool[Args, Result]: ...


class PartialToolWrapper[Result](Protocol):
    def __call__[**Args](
        self,
        function: Callable[Args, Coroutine[Any, Any, Result]],
    ) -> Tool[Args, Result]: ...


@overload
def tool[**Args, Result](
    function: Callable[Args, Coroutine[Any, Any, Result]],
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
    format_result: Callable[[Result], MultimodalContent | MultimodalContentConvertible],
    format_failure: Callable[[Exception], MultimodalContent | MultimodalContentConvertible]
    | None = None,
    direct_result: bool = False,
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
    format_failure: Callable[[Exception], MultimodalContent | MultimodalContentConvertible]
    | None = None,
    direct_result: bool = False,
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

    Returns
    -------
    Callable[[Function[Args, Coroutine[None, None, Result]]], Tool[Args, Result]]
        function allowing to convert other function to a Tool using provided configuration.
    """


def tool[**Args, Result](  # noqa: PLR0913
    function: Callable[Args, Coroutine[Any, Any, Result]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability_check: ToolAvailabilityCheck | None = None,
    format_result: Callable[[Result], MultimodalContent | MultimodalContentConvertible]
    | None = None,
    format_failure: Callable[[Exception], MultimodalContent | MultimodalContentConvertible]
    | None = None,
    direct_result: bool = False,
) -> ToolWrapper | Tool[Args, Result]:
    def wrap(
        function: Callable[Args, Coroutine[Any, Any, Result]],
    ) -> Tool[Args, Result]:
        return Tool(
            name=name or function.__name__,
            description=description,
            function=function,
            availability_check=availability_check,
            format_result=format_result or _default_result_format,
            format_failure=format_failure or _default_failure_result,
            direct_result=direct_result,
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

        case other:
            return MultimodalContent.of(str(other))


def _default_failure_result(exception: Exception) -> MultimodalContent:
    ctx.log_error("Tool call failure", exception=exception)
    return MultimodalContent.of("ERROR")
