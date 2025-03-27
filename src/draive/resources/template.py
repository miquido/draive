from collections.abc import Callable, Coroutine, Sequence
from typing import Protocol, final

from draive.commons import META_EMPTY, Meta
from draive.parameters import ParametrizedFunction
from draive.resources.types import (
    Resource,
    ResourceContent,
    ResourceDeclaration,
    ResourceException,
)

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
    __slots__ = (
        "_check_availability",
        "declaration",
        "uri",
    )

    def __init__(  # noqa: PLR0913
        self,
        /,
        uri: str,
        *,
        mime_type: str | None,
        name: str,
        description: str | None,
        availability_check: ResourceAvailabilityCheck | None,
        meta: Meta,
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> None:
        super().__init__(function)

        self.uri: str = uri
        self.declaration: ResourceDeclaration = ResourceDeclaration(
            uri=uri,
            mime_type=mime_type,
            name=name,
            description=description,
            meta=meta,
        )
        self._check_availability: ResourceAvailabilityCheck = availability_check or (
            lambda: True  # available by default
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
    ) -> Resource:
        try:
            return Resource(
                name=self.declaration.name,
                description=self.declaration.description,
                uri=self.uri,
                content=await super().__call__(*args, **kwargs),  # pyright: ignore[reportCallIssue],
                meta=self.declaration.meta,
            )

        except Exception as exc:
            raise ResourceException(f"Resolving resource '{self.declaration.uri}' failed") from exc


def resource[**Args, Result: Sequence[Resource] | ResourceContent](  # noqa: PLR0913
    *,
    uri: str,
    mime_type: str | None = None,
    name: str | None = None,
    description: str | None = None,
    availability_check: ResourceAvailabilityCheck | None = None,
    meta: Meta | None = None,
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
            meta=meta if meta is not None else META_EMPTY,
        )

    return wrap
