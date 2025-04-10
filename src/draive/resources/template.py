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

__all__ = (
    "ResourceAvailabilityCheck",
    "ResourceTemplate",
    "resource",
)


class ResourceAvailabilityCheck(Protocol):
    def __call__(self) -> bool: ...


# TODO: add https://modelcontextprotocol.io/docs/concepts/resources#resource-templates
# support and uri template resolving
@final
class ResourceTemplate[**Args](
    ParametrizedFunction[Args, Coroutine[None, None, Sequence[Resource] | ResourceContent]]
):
    __slots__ = (
        "_check_availability",
        "declaration",
        "uri",
    )

    def __init__(
        self,
        /,
        uri: str,
        *,
        mime_type: str | None,
        name: str,
        description: str | None,
        availability_check: ResourceAvailabilityCheck | None,
        meta: Meta,
        function: Callable[Args, Coroutine[None, None, Sequence[Resource] | ResourceContent]],
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
                content=await super().__call__(*args, **kwargs),
                meta=self.declaration.meta,
            )

        except Exception as exc:
            raise ResourceException(f"Resolving resource '{self.declaration.uri}' failed") from exc

    async def resolve_from_uri(
        self,
        uri: str,
        /,
    ) -> Resource:
        # TODO: resolve args from uri - it will now work only for resources without args
        try:
            return Resource(
                name=self.declaration.name,
                description=self.declaration.description,
                uri=self.uri,
                # call args will be validated
                content=await super().__call__(),  # pyright: ignore[reportCallIssue]
                meta=self.declaration.meta,
            )

        except Exception as exc:
            raise ResourceException(f"Resolving resource '{self.declaration.uri}' failed") from exc


def resource[**Args](
    *,
    uri: str,
    mime_type: str | None = None,
    name: str | None = None,
    description: str | None = None,
    availability_check: ResourceAvailabilityCheck | None = None,
    meta: Meta | None = None,
) -> Callable[
    [Callable[Args, Coroutine[None, None, Sequence[Resource] | ResourceContent]]],
    ResourceTemplate[Args],
]:
    def wrap(
        function: Callable[Args, Coroutine[None, None, Sequence[Resource] | ResourceContent]],
    ) -> ResourceTemplate[Args]:
        return ResourceTemplate[Args](
            uri,
            mime_type=mime_type,
            name=name or function.__name__,
            description=description,
            availability_check=availability_check,
            function=function,
            meta=meta if meta is not None else META_EMPTY,
        )

    return wrap
