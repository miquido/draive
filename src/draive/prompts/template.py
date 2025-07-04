from collections.abc import Callable, Coroutine
from typing import Protocol, final, overload

from haiway import ctx

from draive.commons import Meta, MetaValues
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
        meta: Meta,
        function: Callable[Args, Coroutine[None, None, LMMContext]],
    ) -> None:
        super().__init__(function)

        self.declaration: PromptDeclaration
        object.__setattr__(
            self,
            "declaration",
            PromptDeclaration(
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
                meta=meta,
            ),
        )
        self._check_availability: PromptAvailabilityCheck
        object.__setattr__(
            self,
            "_check_availability",
            availability_check
            or (
                lambda meta: True  # available by default
            ),
        )

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
            try:
                result = Prompt(
                    name=self.declaration.name,
                    description=self.declaration.description,
                    content=await super().__call__(*args, **kwargs),  # pyright: ignore[reportCallIssue]
                )

                return result

            except BaseException as exc:
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
def prompt(
    *,
    name: str | None = None,
    description: str | None = None,
    availability_check: PromptAvailabilityCheck | None = None,
    meta: Meta | MetaValues | None = None,
) -> PromptTemplateWrapper: ...


def prompt[**Args](
    function: Callable[Args, Coroutine[None, None, LMMContext]] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    availability_check: PromptAvailabilityCheck | None = None,
    meta: Meta | MetaValues | None = None,
) -> PromptTemplateWrapper | PromptTemplate[Args]:
    def wrap[**Arg](
        function: Callable[Arg, Coroutine[None, None, LMMContext]],
    ) -> PromptTemplate[Arg]:
        return PromptTemplate[Arg](
            name=name or function.__name__,
            description=description,
            availability_check=availability_check,
            meta=Meta.of(meta),
            function=function,
        )

    if function := function:
        return wrap(function=function)

    else:
        return wrap
