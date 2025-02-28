from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self, runtime_checkable

from haiway import State

from draive.lmm import LMMContext, LMMContextElement
from draive.parameters import DataModel, Field
from draive.parameters.specification import ParameterSpecification

__all__ = [
    "MissingPrompt",
    "Prompt",
    "PromptDeclaration",
    "PromptDeclarationArgument",
    "PromptFetching",
    "PromptListing",
]


class MissingPrompt(Exception):
    pass


class PromptDeclarationArgument(DataModel):
    name: str
    specification: ParameterSpecification = Field(
        specification={
            "type": "object",
            "additionalProperties": True,
        }
    )
    required: bool = True


class PromptDeclaration(DataModel):
    name: str
    description: str | None = None
    arguments: Sequence[PromptDeclarationArgument]
    meta: Mapping[str, str | float | int | bool | None] | None


class Prompt(State):
    @classmethod
    def of(
        cls,
        *content: LMMContextElement,
        name: str,
        description: str | None = None,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls(
            name=name,
            description=description,
            content=content,
            meta=meta,
        )

    name: str
    description: str | None = None
    content: LMMContext
    meta: Mapping[str, str | float | int | bool | None] | None


@runtime_checkable
class PromptListing(Protocol):
    async def __call__(
        self,
        **extra: Any,
    ) -> Sequence[PromptDeclaration]: ...


@runtime_checkable
class PromptFetching(Protocol):
    async def __call__(
        self,
        name: str,
        *,
        arguments: Mapping[str, str] | None,
        **extra: Any,
    ) -> Prompt | None: ...
