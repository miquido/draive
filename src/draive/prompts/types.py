from collections.abc import Mapping, Sequence
from typing import Protocol, Self, runtime_checkable

from draive.lmm import LMMContext, LMMContextElement
from draive.parameters import DataModel

__all__ = [
    "Prompt",
    "PromptDeclaration",
    "PromptDeclarationArgument",
    "PromptFetching",
    "PromptListing",
]


class PromptDeclarationArgument(DataModel):
    name: str
    description: str | None = None
    required: bool = True


class PromptDeclaration(DataModel):
    name: str
    description: str | None = None
    arguments: Sequence[PromptDeclarationArgument]


class Prompt(DataModel):
    @classmethod
    def of(
        cls,
        name: str,
        /,
        *content: LMMContextElement,
        description: str | None = None,
    ) -> Self:
        return cls(
            name=name,
            description=description,
            content=content,
        )

    name: str
    description: str | None = None
    content: LMMContext


@runtime_checkable
class PromptListing(Protocol):
    async def __call__(
        self,
    ) -> Sequence[PromptDeclaration]: ...


@runtime_checkable
class PromptFetching(Protocol):
    async def __call__(
        self,
        name: str,
        *,
        arguments: Mapping[str, str] | None,
    ) -> Prompt: ...
