from collections.abc import Callable, Coroutine
from typing import Protocol, final, overload

from haiway import Meta, MetaValues

from draive.models.instructions.types import (
    InstructionsArgumentDeclaration,
    InstructionsDeclaration,
)
from draive.parameters import ParametrizedFunction

__all__ = (
    "InstructionsTemplate",
    "instructions",
)


@final
class InstructionsTemplate[**Args](ParametrizedFunction[Args, Coroutine[None, None, str]]):
    """Wrap an async function to define instructions content and declaration.

    Parameters
    ----------
    function : Callable[..., Coroutine[None, None, str]]
        Async function producing instructions content when invoked.
    name : str
        Name assigned to the resulting instructions template.
    description : str | None
        Optional human-readable description of the template.
    meta : Meta
        Metadata stored on the resulting declaration.

    Attributes
    ----------
    declaration : InstructionsDeclaration
        Declaration derived from the wrapped function signature.
    """

    __slots__ = ("declaration",)

    def __init__(
        self,
        /,
        function: Callable[Args, Coroutine[None, None, str]],
        *,
        name: str,
        description: str | None,
        meta: Meta,
    ) -> None:
        super().__init__(function)

        self.declaration: InstructionsDeclaration
        object.__setattr__(
            self,
            "declaration",
            InstructionsDeclaration(
                name=name,
                description=description,
                arguments=[  # should we verify if all are strings?
                    InstructionsArgumentDeclaration(
                        name=parameter.alias or parameter.name,
                        description=parameter.description,
                        required=parameter.required,
                    )
                    for parameter in self._parameters.values()
                ],
                meta=meta,
            ),
        )


class InstructionsTemplateWrapper(Protocol):
    def __call__[**Args](
        self,
        function: Callable[Args, Coroutine[None, None, str]],
    ) -> InstructionsTemplate[Args]: ...


@overload
def instructions[**Args](
    function: Callable[Args, Coroutine[None, None, str]],
    /,
) -> InstructionsTemplate[Args]:
    """Decorator: convert a function into an ``InstructionsTemplate`` using defaults."""


@overload
def instructions(
    *,
    name: str | None = None,
    description: str | None = None,
    meta: Meta | MetaValues | None = None,
) -> InstructionsTemplateWrapper:
    """Decorator factory for building ``InstructionsTemplate`` with parameters."""


def instructions[**Args](
    function: Callable[Args, Coroutine[None, None, str]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    meta: Meta | MetaValues | None = None,
) -> InstructionsTemplateWrapper | InstructionsTemplate[Args]:
    """Convert a function into an ``InstructionsTemplate``.

    Parameters
    ----------
    function : Callable[..., Coroutine[None, None, str]] | None, optional
        Async function producing template content. When ``None`` a decorator is
        produced awaiting the function.
    name : str | None, optional
        Explicit name for the template. Defaults to the wrapped function name.
    description : str | None, optional
        Optional human-readable description.
    meta : Meta | MetaValues | None, optional
        Metadata merged into the template declaration.

    Returns
    -------
    InstructionsTemplate | InstructionsTemplateWrapper
        ``InstructionsTemplate`` when a function is provided, otherwise a decorator
        configuring template creation.
    """

    def wrap[**Arg](
        function: Callable[Arg, Coroutine[None, None, str]],
    ) -> InstructionsTemplate[Arg]:
        return InstructionsTemplate[Arg](
            function=function,
            name=name or function.__name__,
            description=description,
            meta=Meta.of(meta),
        )

    if function is not None:
        return wrap(function=function)

    else:
        return wrap
