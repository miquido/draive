from collections.abc import Mapping, Sequence
from typing import Any, Protocol, Self, runtime_checkable

from haiway import State

from draive.commons import META_EMPTY, Meta, MetaValues
from draive.lmm import LMMContext, LMMContextElement
from draive.parameters import DataModel, Field
from draive.parameters.specification import ParameterSpecification

__all__ = (
    "Prompt",
    "PromptDeclaration",
    "PromptDeclarationArgument",
    "PromptException",
    "PromptFetching",
    "PromptListFetching",
    "PromptMissing",
)


class PromptException(Exception):
    pass


class PromptMissing(PromptException):
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
    meta: Meta = META_EMPTY


class Prompt(State):
    @classmethod
    def of(
        cls,
        *content: LMMContextElement,
        name: str,
        description: str | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            name=name,
            description=description,
            content=content,
            meta=Meta.of(meta),
        )

    name: str
    description: str | None = None
    content: LMMContext
    meta: Meta = META_EMPTY


@runtime_checkable
class PromptListFetching(Protocol):
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
