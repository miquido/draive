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
    """Wraps an async function to define instructions content and declaration.

    Builds an ``InstructionsDeclaration`` from the function's parameter specifications
    and exposes a callable that renders the content at runtime.
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

    When called without a function, returns a decorator with configured parameters.
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
