from collections.abc import Callable, Coroutine, Sequence
from typing import Protocol, final

from haiway import ArgumentsTrace, ResultTrace, ctx, freeze

from draive.parameters import ParametrizedFunction
from draive.resources.types import Resource, ResourceContent, ResourceDeclaration

__all__ = [
    "ResourceAvailabilityCheck",
    "ResourceTemplate",
    "resource",
]


class ResourceAvailabilityCheck(Protocol):
    def __call__(self) -> bool: ...


# TODO: add https://modelcontextprotocol.io/docs/concepts/resources#resource-templates
# support and uri template resolving
@final
class ResourceTemplate[**Args, Result: Sequence[Resource] | ResourceContent](
    ParametrizedFunction[Args, Coroutine[None, None, Result]]
):
    def __init__(  # noqa: PLR0913
        self,
        /,
        uri: str,
        *,
        mime_type: str | None,
        name: str,
        description: str | None,
        availability_check: ResourceAvailabilityCheck | None = None,
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> None:
        super().__init__(function)

        self.uri: str = uri
        self.declaration: ResourceDeclaration = ResourceDeclaration(
            uri=uri,
            mime_type=mime_type,
            name=name,
            description=description,
        )
        self._check_availability: ResourceAvailabilityCheck = availability_check or (
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
    ) -> Resource:
        with ctx.scope(f"resource:{self.uri}"):
            ctx.record(ArgumentsTrace.of(*args, **kwargs))
            try:
                result = Resource(
                    uri=self.uri,
                    content=await super().__call__(*args, **kwargs),  # pyright: ignore[reportCallIssue],
                )
                ctx.record(ResultTrace.of(result))

                return result

            except BaseException as exc:
                ctx.record(ResultTrace.of(exc))
                ctx.log_error(
                    "Resource resolving error",
                    exception=exc,
                )
                raise exc


def resource[**Args, Result: Sequence[Resource] | ResourceContent](
    *,
    uri: str,
    mime_type: str | None = None,
    name: str | None = None,
    description: str | None = None,
    availability_check: ResourceAvailabilityCheck | None = None,
) -> Callable[
    [Callable[Args, Coroutine[None, None, Result]]],
    ResourceTemplate[Args, Result],
]:
    def wrap(
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> ResourceTemplate[Args, Result]:
        return ResourceTemplate[Args, Result](
            uri,
            mime_type=mime_type,
            name=name or function.__name__,
            description=description,
            availability_check=availability_check,
            function=function,
        )

    return wrap
