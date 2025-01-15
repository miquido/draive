from collections.abc import Callable, Coroutine
from typing import final, overload

from haiway import ArgumentsTrace, ResultTrace, ctx, freeze

from draive.instructions.types import (
    Instruction,
    InstructionDeclaration,
    InstructionDeclarationArgument,
)
from draive.parameters import ParametrizedFunction

__all__ = [
    "InstructionTemplate",
    "instruction",
]


@final
class InstructionTemplate[**Args](ParametrizedFunction[Args, Coroutine[None, None, str]]):
    def __init__(
        self,
        /,
        name: str,
        *,
        description: str | None = None,
        function: Callable[Args, Coroutine[None, None, str]],
    ) -> None:
        super().__init__(function)

        self.declaration: InstructionDeclaration = InstructionDeclaration(
            name=name,
            description=description,
            arguments=[
                InstructionDeclarationArgument(
                    name=parameter.alias or parameter.name,
                    specification=parameter.specification,
                    required=parameter.required,
                )
                for parameter in self._parameters.values()
            ],
        )

        freeze(self)

    async def resolve(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Instruction:
        with ctx.scope(f"instruction:{self.declaration.name}"):
            ctx.record(ArgumentsTrace.of(*args, **kwargs))
            try:
                result = Instruction(
                    name=self.declaration.name,
                    description=self.declaration.description,
                    content=await super().__call__(*args, **kwargs),  # pyright: ignore[reportCallIssue]
                )
                ctx.record(ResultTrace.of(result))

                return result

            except BaseException as exc:
                ctx.record(ResultTrace.of(exc))
                ctx.log_error(
                    "Instruction resolving error",
                    exception=exc,
                )
                raise exc


@overload
def instruction[**Args](
    function: Callable[Args, Coroutine[None, None, str]],
    /,
) -> Callable[[Callable[Args, Coroutine[None, None, str]]], InstructionTemplate[Args]]: ...


@overload
def instruction[**Args](
    *,
    name: str | None = None,
    description: str | None = None,
) -> InstructionTemplate[Args]: ...


def instruction[**Args](
    function: Callable[Args, Coroutine[None, None, str]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> (
    Callable[[Callable[Args, Coroutine[None, None, str]]], InstructionTemplate[Args]]
    | InstructionTemplate[Args]
):
    def wrap(
        function: Callable[Args, Coroutine[None, None, str]],
    ) -> InstructionTemplate[Args]:
        return InstructionTemplate[Args](
            name=name or function.__name__,
            description=description,
            function=function,
        )

    if function := function:
        return wrap(function=function)

    else:
        return wrap
