from collections.abc import Callable, Coroutine
from typing import Protocol, final, overload

from haiway import ArgumentsTrace, ResultTrace, ctx, freeze

from draive.commons import META_EMPTY, Meta
from draive.lmm import LMMContext
from draive.parameters import ParametrizedFunction
from draive.prompts.types import Prompt, PromptDeclaration, PromptDeclarationArgument

__all__ = (
    "PromptAvailabilityCheck",
    "PromptTemplate",
    "prompt",
)


class PromptAvailabilityCheck(Protocol):
    def __call__(self) -> bool: ...


@final
class PromptTemplate[**Args](ParametrizedFunction[Args, Coroutine[None, None, LMMContext]]):
    __slots__ = ("_check_availability", "declaration")

    def __init__(
        self,
        /,
        name: str,
        *,
        description: str | None = None,
        availability_check: PromptAvailabilityCheck | None,
        meta: Meta | None,
        function: Callable[Args, Coroutine[None, None, LMMContext]],
    ) -> None:
        super().__init__(function)

        self.declaration: PromptDeclaration = PromptDeclaration(
            name=name,
            description=description,
            arguments=[
                PromptDeclarationArgument(
                    name=parameter.alias or parameter.name,
                    specification=parameter.specification,
                    required=parameter.required,
                )
                for parameter in self._parameters.values()
            ],
            meta=meta if meta is not None else META_EMPTY,
        )
        self._check_availability: PromptAvailabilityCheck = availability_check or (
            lambda: True  # available by default
        )

        freeze(self)

    @property
    def available(self) -> bool:
        try:
            return self._check_availability()

        except Exception:
            return False

    async def resolve(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Prompt:
        with ctx.scope(f"prompt:{self.declaration.name}"):
            ctx.record(ArgumentsTrace.of(*args, **kwargs))
            try:
                result = Prompt(
                    name=self.declaration.name,
                    description=self.declaration.description,
                    content=await super().__call__(*args, **kwargs),  # pyright: ignore[reportCallIssue]
                )
                ctx.record(ResultTrace.of(result))

                return result

            except BaseException as exc:
                ctx.record(ResultTrace.of(exc))
                ctx.log_error(
                    "Prompt resolving error",
                    exception=exc,
                )
                raise exc


class PromptTemplateWrapper(Protocol):
    def __call__[**Args](
        self,
        function: Callable[Args, Coroutine[None, None, LMMContext]],
    ) -> PromptTemplate[Args]: ...


@overload
def prompt[**Args](
    function: Callable[Args, Coroutine[None, None, LMMContext]],
    /,
) -> PromptTemplate[Args]: ...


@overload
def prompt[**Args](
    *,
    name: str | None = None,
    description: str | None = None,
    availability_check: PromptAvailabilityCheck | None = None,
    meta: Meta | None = None,
) -> PromptTemplateWrapper: ...


def prompt[**Args](
    function: Callable[Args, Coroutine[None, None, LMMContext]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability_check: PromptAvailabilityCheck | None = None,
    meta: Meta | None = None,
) -> PromptTemplateWrapper | PromptTemplate[Args]:
    def wrap[**Arg](
        function: Callable[Arg, Coroutine[None, None, LMMContext]],
    ) -> PromptTemplate[Arg]:
        return PromptTemplate[Arg](
            name=name or function.__name__,
            description=description,
            availability_check=availability_check,
            meta=meta,
            function=function,
        )

    if function := function:
        return wrap(function=function)

    else:
        return wrap
