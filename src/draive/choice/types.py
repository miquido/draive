from collections.abc import Iterable, Sequence
from typing import Any, Protocol, Self, runtime_checkable
from uuid import uuid4

from haiway import State

from draive.commons import Meta, MetaValue
from draive.instructions import Instruction
from draive.multimodal import Multimodal, MultimodalContent
from draive.tools import Toolbox

__all__ = (
    "ChoiceCompletion",
    "ChoiceOption",
    "SelectionException",
)


class ChoiceOption(State):
    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        identifier: str | None = None,
        **meta: MetaValue,
    ) -> Self:
        assert identifier is None or len(identifier) > 0, "Identifier can't be empty"  # nosec: B101

        return cls(
            identifier=identifier or uuid4().hex,
            content=MultimodalContent.of(content),
            meta=Meta.of(meta),
        )

    identifier: str
    content: MultimodalContent
    meta: Meta


@runtime_checkable
class ChoiceCompletion(Protocol):
    async def __call__(
        self,
        *,
        instruction: Instruction | str,
        options: Sequence[ChoiceOption],
        input: Multimodal,  # noqa: A002
        toolbox: Toolbox,
        examples: Iterable[tuple[Multimodal, ChoiceOption]] | None,
        **extra: Any,
    ) -> ChoiceOption: ...


class SelectionException(Exception):
    pass
