from collections.abc import Callable, Coroutine, Iterable
from typing import Any, Protocol, Self, cast, final, overload

from haiway import Meta, MetaValues, TypeSpecification, ctx

from draive.models.tools.types import (
    ToolAvailabilityChecking,
    ToolError,
    ToolErrorFormatting,
    ToolResultFormatting,
)
from draive.models.types import (
    ModelToolFunctionSpecification,
    ModelToolHandling,
    ModelToolParametersSpecification,
    ModelToolSpecification,
)
from draive.multimodal import (
    Multimodal,
    MultimodalContent,
)
from draive.parameters import ParametrizedFunction

__all__ = (
    "FunctionTool",
    "tool",
)


@final
class FunctionTool[**Args, Result](ParametrizedFunction[Args, Coroutine[None, None, Result]]):
    """Wraps an async function and exposes it as a Tool.

    Builds the tool specification from the function signature (or provided parameters),
    handles availability checks, formats results and errors into ``MultimodalContent``,
    and supports both tool-style invocation and regular function calls.

    Attributes
    ----------
    name : str
        Tool name.
    description : str | None
        Human-readable tool description.
    parameters : ModelToolParametersSpecification | None
        JSON Schema-like parameters spec as provided or inferred from function arguments.
    specification : ModelToolSpecification
        Full tool specification exposed to the model.
    handling : ModelToolHandling
        Response handling policy (e.g., ``"response"``, ``"output"``).
    """

    __slots__ = (
        "_check_availability",
        "description",
        "format_error",
        "format_result",
        "handling",
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
        parameters: ModelToolParametersSpecification | None,
        availability: ToolAvailabilityChecking,
        result_formatting: ToolResultFormatting[Result],
        error_formatting: ToolErrorFormatting,
        handling: ModelToolHandling = "response",
        meta: Meta,
    ) -> None:
        super().__init__(function)
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

        if parameters is None:
            aliased_required: list[str] = []
            specifications: dict[str, TypeSpecification] = {}
            for parameter in self._parameters.values():
                specification: TypeSpecification | None = parameter.specification
                if specification is None:
                    raise RuntimeError(
                        f"Function argument {parameter.name} does not provide a valid specification"
                    )

                specifications[parameter.alias or parameter.name] = specification

                if parameter.required:
                    aliased_required.append(parameter.alias or parameter.name)

            parameters = {
                "type": "object",
                "properties": specifications,
                "required": aliased_required,
                "additionalProperties": False,
            }

        self.parameters: ModelToolParametersSpecification
        object.__setattr__(
            self,
            "parameters",
            parameters,
        )
        self.specification: ModelToolSpecification
        object.__setattr__(
            self,
            "specification",
            ModelToolFunctionSpecification(
                name=name,
                description=description,
                parameters=parameters,
                meta=meta,
            ),
        )
        self.handling: ModelToolHandling
        object.__setattr__(
            self,
            "handling",
            handling,
        )
        self._check_availability: ToolAvailabilityChecking
        object.__setattr__(
            self,
            "_check_availability",
            availability,
        )
        self.format_result: ToolResultFormatting[Result]
        object.__setattr__(
            self,
            "format_result",
            result_formatting,
        )
        self.format_error: ToolErrorFormatting
        object.__setattr__(
            self,
            "format_error",
            error_formatting,
        )

    @property
    def meta(self) -> Meta:
        return self.specification.meta

    def updated(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        function: Callable[Args, Coroutine[None, None, Result]] | None = None,
        parameters: ModelToolParametersSpecification | None = None,
        availability: ToolAvailabilityChecking | None = None,
        result_formatting: ToolResultFormatting[Result] | None = None,
        error_formatting: ToolErrorFormatting | None = None,
        handling: ModelToolHandling | None = None,
        meta: Meta | None = None,
    ) -> Self:
        """Return a new tool with updated configuration.

        Parameters
        ----------
        name : str | None, optional
            Override for the tool name.
        description : str | None, optional
            Override for the tool description.
        function : Callable[Args, Coroutine[None, None, Result]] | None, optional
            Replacement callable backing the tool.
        parameters : ModelToolParametersSpecification | None, optional
            Replacement parameters specification. ``None`` keeps existing value.
        availability : ToolAvailabilityChecking | None, optional
            Custom availability checker; defaults to current checker.
        result_formatting : ToolResultFormatting[Result] | None, optional
            Replacement result formatter.
        error_formatting : ToolErrorFormatting | None, optional
            Replacement error formatter.
        handling : ModelToolHandling | None, optional
            Updated handling strategy for the model response.
        meta : Meta | None, optional
            Replacement metadata for the tool.

        Returns
        -------
        FunctionTool
            Newly constructed tool instance with applied overrides.
        """
        return self.__class__(
            function or self._call,
            name=name or self.name,
            description=description if description is not None else self.description,
            parameters=parameters if parameters is not None else self.parameters,
            availability=availability if availability is not None else self._check_availability,
            result_formatting=result_formatting
            if result_formatting is not None
            else self.format_result,
            error_formatting=error_formatting
            if error_formatting is not None
            else self.format_error,
            handling=handling if handling is not None else self.handling,
            meta=meta or self.specification.meta,
        )

    def available(
        self,
        tools_turn: int,
    ) -> bool:
        """Return ``True`` when the tool is available for the given turn.

        Parameters
        ----------
        tools_turn : int
            Index of the current tool turn (0-based).

        Returns
        -------
        bool
            ``True`` when the tool may be used, otherwise ``False``.
        """
        try:
            return self._check_availability(
                tools_turn=tools_turn,
                specification=self.specification,
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
        """Invoke the underlying function as a tool and format the result.

        Errors are caught and wrapped into ``ToolError`` with formatted content; base
        exceptions are re-raised.

        Parameters
        ----------
        call_id : str
            Identifier of the tool invocation.
        **arguments : Any
            Arguments passed to the underlying function.

        Returns
        -------
        MultimodalContent
            Formatted tool result.

        Raises
        ------
        ToolError
            When the wrapped function raises an exception that should be surfaced to the model.
        """
        async with ctx.scope(f"tool.{self.name}"):
            ctx.record_info(
                attributes={"call_id": call_id},
            )

            try:
                result: Result = await super().__call__(**arguments)  # pyright: ignore[reportCallIssue]

                return MultimodalContent.of(self.format_result(result))

            except Exception as exc:
                ctx.record_error(
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
                ctx.record_error(
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
        """Call the underlying function directly, bypassing tool formatting.

        Parameters
        ----------
        *args : Args.args
            Positional arguments for the wrapped function.
        **kwargs : Args.kwargs
            Keyword arguments for the wrapped function.

        Returns
        -------
        Result
            Raw result returned by the wrapped function.
        """
        return await super().__call__(
            *args,
            **kwargs,
        )


class FunctionToolWrapper(Protocol):
    def __call__[**Args, Result](
        self,
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> FunctionTool[Args, Result]: ...


class PartialFunctionToolWrapper[Result](Protocol):
    def __call__[**Args](
        self,
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> FunctionTool[Args, Result]: ...


def _available(
    tools_turn: int,
    specification: ModelToolSpecification,
) -> bool:
    return True


def _default_result_format(result: Any) -> MultimodalContent:
    if isinstance(result, MultimodalContent):
        return result

    elif isinstance(result, Multimodal):
        return MultimodalContent.of(cast(Multimodal, result))

    elif isinstance(result, Iterable):
        return MultimodalContent.of(
            *(
                cast(Multimodal, element) if isinstance(element, Multimodal) else str(element)
                for element in cast(Iterable[Any], result)
            )
        )

    else:
        return MultimodalContent.of(str(result))


def _default_error_format(error: Exception) -> MultimodalContent:
    return MultimodalContent.of(f"ERROR: Tool execution failed - {type(error).__name__}")


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
    FunctionTool[Args, Result]
        a tool representation of the provided function.
    """


@overload
def tool[Result](
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailabilityChecking = _available,
    result_formatting: ToolResultFormatting[Result],
    error_formatting: ToolErrorFormatting = _default_error_format,
    handling: ModelToolHandling = "response",
    meta: Meta | MetaValues | None = None,
) -> PartialFunctionToolWrapper[Result]:
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
    description: str | None
        Description to be used in a tool specification. Allows presenting tool behavior to the
        external system. Default is ``None``.
    availability: ToolAvailabilityChecking
        Function returning ``True`` when the tool should be available in the given context.
        Default always returns ``True``.
    result_formatting: ToolResultFormatting[Result]
        Function converting tool result to ``MultimodalContent``. Default converts results to
        strings when needed.
    error_formatting: ToolErrorFormatting
        Function converting tool call exceptions to fallback ``MultimodalContent``.
        Default returns an error string and logs the exception.
    handling: ModelToolHandling
        controls if tool result should break the ongoing processing and be the direct result of it.
        Note that during concurrent execution of multiple tools the call/result order defines
        direct result and exact behavior is not defined.
        Default is "response".
    meta: Meta | MetaValues | None
        Custom metadata allowing to access tool metadata such as remote source identifiers.

    Returns
    -------
    Callable[[Function[Args, Coroutine[None, None, Result]]], FunctionTool[Args, Result]]
        function allowing to convert other function to a tool using provided configuration.
    """


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailabilityChecking = _available,
    error_formatting: ToolErrorFormatting = _default_error_format,
    handling: ModelToolHandling = "response",
    meta: Meta | MetaValues | None = None,
) -> FunctionToolWrapper:
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
    description: str | None
        Description to be used in a tool specification. Allows presenting tool behavior to the
        external system. Default is ``None``.
    availability: ToolAvailabilityChecking
        Function returning ``True`` when the tool should be available in the given context.
        Default always returns ``True``.
    error_formatting: ToolErrorFormatting
        Function converting tool call exceptions to fallback ``MultimodalContent``.
        Default returns an error string and logs the exception.
    handling: ModelToolHandling
        controls if tool result should break the ongoing processing and be the direct result of it.
        Note that during concurrent execution of multiple tools the call/result order defines
        direct result and exact behavior is not defined.
        Default is "response".
    meta: Meta | MetaValues | None
        Custom metadata allowing to access tool metadata such as remote source identifiers.

    Returns
    -------
    Callable[[Function[Args, Coroutine[None, None, Result]]], FunctionTool[Args, Result]]
        function allowing to convert other function to a tool using provided configuration.
    """


def tool[**Args, Result](
    function: Callable[Args, Coroutine[None, None, Result]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability: ToolAvailabilityChecking = _available,
    result_formatting: ToolResultFormatting[Result] = _default_result_format,
    error_formatting: ToolErrorFormatting = _default_error_format,
    handling: ModelToolHandling = "response",
    meta: Meta | MetaValues | None = None,
) -> PartialFunctionToolWrapper[Result] | FunctionToolWrapper | FunctionTool[Args, Result]:
    def wrap[**Arg](
        function: Callable[Arg, Coroutine[None, None, Result]],
    ) -> FunctionTool[Arg, Result]:
        return FunctionTool[Arg, Result](
            function=function,
            name=name or function.__name__,
            description=description,
            parameters=None,
            availability=availability,
            result_formatting=result_formatting,
            error_formatting=error_formatting,
            handling=handling,
            meta=Meta.of(meta),
        )

    if function is not None:
        return wrap(function=function)

    else:
        return cast(PartialFunctionToolWrapper[Result], wrap)
