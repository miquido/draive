from collections.abc import AsyncIterable, Callable, Coroutine
from inspect import isasyncgenfunction, iscoroutinefunction, unwrap
from typing import Protocol, Self, TypeIs, final, overload

from haiway import BasicValue, Function, Meta, MetaValues, TypeSpecification

from draive.models import (
    ModelToolFunctionSpecification,
    ModelToolHandling,
    ModelToolParametersSpecification,
    ModelToolSpecification,
)
from draive.multimodal import Multimodal, MultimodalContent
from draive.tools.types import ToolOutputChunk

__all__ = (
    "CoroutineTool",
    "GeneratorTool",
    "tool",
)


def _is_coroutine_tool[**Args](
    function: Callable[Args, AsyncIterable[ToolOutputChunk]]
    | Callable[Args, Coroutine[None, None, Multimodal]],
) -> TypeIs[Callable[Args, Coroutine[None, None, Multimodal]]]:
    return iscoroutinefunction(unwrap(function))


def _is_generator_tool[**Args](
    function: Callable[Args, AsyncIterable[ToolOutputChunk]]
    | Callable[Args, Coroutine[None, None, Multimodal]],
) -> TypeIs[Callable[Args, AsyncIterable[ToolOutputChunk]]]:
    return isasyncgenfunction(unwrap(function))


@final
class CoroutineTool[**Args](Function[Args, Coroutine[None, None, Multimodal]]):
    """Tool adapter wrapping an async function returning multimodal content."""

    __slots__ = (
        "description",
        "handling",
        "name",
        "parameters",
        "specification",
    )

    def __init__(
        self,
        /,
        function: Callable[Args, Coroutine[None, None, Multimodal]],
        *,
        name: str,
        description: str | None,
        parameters: ModelToolParametersSpecification | None,
        handling: ModelToolHandling = "response",
        meta: Meta,
    ) -> None:
        """Initialize a coroutine-backed tool.

        Parameters
        ----------
        function : Callable[Args, Coroutine[None, None, Multimodal]]
            Async callable executed when the tool is invoked.
        name : str
            Public tool name exposed to model tool selection.
        description : str | None
            Optional human-readable tool description.
        parameters : ModelToolParametersSpecification | None
            Explicit JSON schema for arguments. When omitted, the schema is inferred
            from the wrapped function signature.
        handling : ModelToolHandling, default="response"
            Output handling mode used by the toolbox.
        meta : Meta
            Metadata attached to the generated tool specification.

        Raises
        ------
        RuntimeError
            If argument schema inference encounters an unsupported function
            parameter without a type specification.
        """
        super().__init__(function)
        self.name: str = name
        self.description: str | None = description
        assert all(arg.name in self._keyword_arguments for arg in self._positional_arguments)  # nosec: B101
        assert self._variadic_positional_arguments is None  # nosec: B101

        if parameters is None:
            aliased_required: list[str] = []
            specifications: dict[str, TypeSpecification] = {}
            for argument in self._keyword_arguments.values():
                specification: TypeSpecification | None = argument.specification
                if specification is None:
                    raise RuntimeError(
                        f"Function argument {argument.name} does not provide a valid specification"
                    )

                specifications[argument.alias or argument.name] = specification

                if argument.required:
                    aliased_required.append(argument.alias or argument.name)

            parameters = {
                "type": "object",
                "properties": specifications,
                "required": aliased_required,
                "additionalProperties": False,
            }

        self.parameters: ModelToolParametersSpecification = parameters
        self.specification: ModelToolSpecification = ModelToolFunctionSpecification(
            name=name,
            description=description,
            parameters=parameters,
            meta=meta,
        )
        self.handling: ModelToolHandling = handling

    @property
    def meta(self) -> Meta:
        """Metadata attached to the underlying model tool specification."""
        return self.specification.meta

    def updating(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        function: Callable[Args, Coroutine[None, None, Multimodal]] | None = None,
        parameters: ModelToolParametersSpecification | None = None,
        handling: ModelToolHandling | None = None,
        meta: Meta | None = None,
    ) -> Self:
        """Return a copy with selected attributes replaced.

        Parameters
        ----------
        name : str | None, default=None
            Replacement tool name.
        description : str | None, default=None
            Replacement description. Passing ``None`` keeps the current value.
        function : Callable[Args, Coroutine[None, None, Multimodal]] | None, default=None
            Replacement callable. When omitted, the current callable is reused.
        parameters : ModelToolParametersSpecification | None, default=None
            Replacement parameter schema. When omitted, the current schema is reused.
        handling : ModelToolHandling | None, default=None
            Replacement handling mode.
        meta : Meta | None, default=None
            Replacement metadata for the tool specification.

        Returns
        -------
        Self
            Updated tool instance.
        """
        return self.__class__(
            function or self._call,
            name=name or self.name,
            description=description if description is not None else self.description,
            parameters=parameters if parameters is not None else self.parameters,
            handling=handling if handling is not None else self.handling,
            meta=meta if meta is not None else self.specification.meta,
        )

    async def call(
        self,
        **arguments: BasicValue,
    ) -> AsyncIterable[ToolOutputChunk]:
        """Execute the wrapped coroutine and stream its content parts.

        Parameters
        ----------
        **arguments : BasicValue
            Tool arguments passed to the wrapped callable.

        Yields
        ------
        ToolOutputChunk
            Content parts produced from the multimodal return value.
        """
        for part in MultimodalContent.of(
            await super().__call__(**arguments)  # pyright: ignore[reportCallIssue]
        ).parts:
            yield part


@final
class GeneratorTool[**Args](Function[Args, AsyncIterable[ToolOutputChunk]]):
    """Tool adapter wrapping an async generator of tool output chunks."""

    __slots__ = (
        "description",
        "handling",
        "name",
        "parameters",
        "specification",
    )

    def __init__(
        self,
        /,
        function: Callable[Args, AsyncIterable[ToolOutputChunk]],
        *,
        name: str,
        description: str | None,
        parameters: ModelToolParametersSpecification | None,
        handling: ModelToolHandling = "response",
        meta: Meta,
    ) -> None:
        """Initialize a generator-backed tool.

        Parameters
        ----------
        function : Callable[Args, AsyncIterable[ToolOutputChunk]]
            Async generator callable executed when the tool is invoked.
        name : str
            Public tool name exposed to model tool selection.
        description : str | None
            Optional human-readable tool description.
        parameters : ModelToolParametersSpecification | None
            Explicit JSON schema for arguments. When omitted, the schema is inferred
            from the wrapped function signature.
        handling : ModelToolHandling, default="response"
            Output handling mode used by the toolbox.
        meta : Meta
            Metadata attached to the generated tool specification.

        Raises
        ------
        RuntimeError
            If argument schema inference encounters an unsupported function
            parameter without a type specification.
        """
        super().__init__(function)
        self.name: str = name
        self.description: str | None = description
        assert all(arg.name in self._keyword_arguments for arg in self._positional_arguments)  # nosec: B101
        assert self._variadic_positional_arguments is None  # nosec: B101

        if parameters is None:
            aliased_required: list[str] = []
            specifications: dict[str, TypeSpecification] = {}
            for argument in self._keyword_arguments.values():
                specification: TypeSpecification | None = argument.specification
                if specification is None:
                    raise RuntimeError(
                        f"Function argument {argument.name} does not provide a valid specification"
                    )

                specifications[argument.alias or argument.name] = specification

                if argument.required:
                    aliased_required.append(argument.alias or argument.name)

            parameters = {
                "type": "object",
                "properties": specifications,
                "required": aliased_required,
                "additionalProperties": False,
            }

        self.parameters: ModelToolParametersSpecification = parameters
        self.specification: ModelToolSpecification = ModelToolFunctionSpecification(
            name=name,
            description=description,
            parameters=parameters,
            meta=meta,
        )
        self.handling: ModelToolHandling = handling

    @property
    def meta(self) -> Meta:
        """Metadata attached to the underlying model tool specification."""
        return self.specification.meta

    def updating(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        function: Callable[Args, AsyncIterable[ToolOutputChunk]] | None = None,
        parameters: ModelToolParametersSpecification | None = None,
        handling: ModelToolHandling | None = None,
        meta: Meta | None = None,
    ) -> Self:
        """Return a copy with selected attributes replaced.

        Parameters
        ----------
        name : str | None, default=None
            Replacement tool name.
        description : str | None, default=None
            Replacement description. Passing ``None`` keeps the current value.
        function : Callable[Args, AsyncIterable[ToolOutputChunk]] | None, default=None
            Replacement callable. When omitted, the current callable is reused.
        parameters : ModelToolParametersSpecification | None, default=None
            Replacement parameter schema. When omitted, the current schema is reused.
        handling : ModelToolHandling | None, default=None
            Replacement handling mode.
        meta : Meta | None, default=None
            Replacement metadata for the tool specification.

        Returns
        -------
        Self
            Updated tool instance.
        """
        return self.__class__(
            function or self._call,
            name=name or self.name,
            description=description if description is not None else self.description,
            parameters=parameters if parameters is not None else self.parameters,
            handling=handling if handling is not None else self.handling,
            meta=meta if meta is not None else self.specification.meta,
        )

    def call(
        self,
        **arguments: BasicValue,
    ) -> AsyncIterable[ToolOutputChunk]:
        """Execute the wrapped async generator.

        Parameters
        ----------
        **arguments : BasicValue
            Tool arguments passed to the wrapped callable.

        Returns
        -------
        AsyncIterable[ToolOutputChunk]
            Original stream emitted by the wrapped tool function.
        """
        return super().__call__(**arguments)  # pyright: ignore[reportCallIssue]


class FunctionToolWrapper(Protocol):
    """Decorator signature for wrapping a function into a tool adapter."""

    def __call__[**Args](
        self,
        function: Callable[Args, AsyncIterable[ToolOutputChunk]]
        | Callable[Args, Coroutine[None, None, Multimodal]],
    ) -> CoroutineTool[Args] | GeneratorTool[Args]:
        """Wrap a single async callable as a configured tool."""
        ...


@overload
def tool[**Args](
    function: Callable[Args, AsyncIterable[ToolOutputChunk]]
    | Callable[Args, Coroutine[None, None, Multimodal]],
    /,
) -> CoroutineTool[Args] | GeneratorTool[Args]:
    """Wrap an async callable into a tool adapter with default settings.

    Parameters
    ----------
    function : Callable[Args, AsyncIterable[ToolOutputChunk]]
        | Callable[Args, Coroutine[None, None, Multimodal]]
        Async callable to expose as a tool.

    Returns
    -------
    CoroutineTool[Args] | GeneratorTool[Args]
        Tool instance with inferred name and inferred parameter schema.
    """


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: ModelToolParametersSpecification | None = None,
    handling: ModelToolHandling = "response",
    meta: Meta | MetaValues | None = None,
) -> FunctionToolWrapper: ...


def tool[**Args](
    function: Callable[Args, AsyncIterable[ToolOutputChunk]]
    | Callable[Args, Coroutine[None, None, Multimodal]]
    | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: ModelToolParametersSpecification | None = None,
    handling: ModelToolHandling = "response",
    meta: Meta | MetaValues | None = None,
) -> FunctionToolWrapper | CoroutineTool[Args] | GeneratorTool[Args]:
    """Create or configure a decorator that turns an async callable into a tool.

    Parameters
    ----------
    function : Callable[Args, AsyncIterable[ToolOutputChunk]]
        | Callable[Args, Coroutine[None, None, Multimodal]]
        | None, default=None
        Callable to wrap immediately. When omitted, a configured decorator is
        returned instead.
    name : str | None, default=None
        Explicit tool name. Defaults to the wrapped function name.
    description : str | None, default=None
        Optional human-readable tool description.
    parameters : ModelToolParametersSpecification | None, default=None
        Explicit JSON schema for arguments. When omitted, the schema is inferred
        from the wrapped callable signature.
    handling : ModelToolHandling, default="response"
        Output handling mode used by the toolbox.
    meta : Meta | MetaValues | None, default=None
        Metadata attached to the generated tool specification.

    Returns
    -------
    FunctionToolWrapper | CoroutineTool[Args] | GeneratorTool[Args]
        Configured decorator or wrapped tool instance, depending on whether
        ``function`` is provided.

    Raises
    ------
    TypeError
        If the provided callable is neither an async coroutine function nor an
        async generator function.
    """

    def wrap[**Arg](
        function: Callable[Arg, AsyncIterable[ToolOutputChunk]]
        | Callable[Arg, Coroutine[None, None, Multimodal]],
    ) -> CoroutineTool[Arg] | GeneratorTool[Arg]:
        if _is_coroutine_tool(function):
            return CoroutineTool[Arg](
                function=function,
                name=name or function.__name__,
                description=description,
                parameters=parameters,
                handling=handling,
                meta=Meta.of(meta),
            )

        if _is_generator_tool(function):
            return GeneratorTool[Arg](
                function=function,
                name=name or function.__name__,
                description=description,
                parameters=parameters,
                handling=handling,
                meta=Meta.of(meta),
            )

        raise TypeError("Unsupported tool function")

    if function is not None:
        return wrap(function=function)

    else:
        return wrap
