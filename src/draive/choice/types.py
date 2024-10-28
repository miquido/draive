from collections.abc import Iterable, Sequence
from typing import Any, Protocol, Self, runtime_checkable
from uuid import uuid4

from haiway import State

from draive.instructions import Instruction
from draive.lmm import Toolbox
from draive.types import Multimodal, MultimodalContent

__all__ = [
    "ChoiceOption",
    "ChoiceCompletion",
    "SelectionException",
]


class ChoiceOption(State):
    @classmethod
    def of(
        cls,
        content: Multimodal,
        /,
        identifier: str | None = None,
        **meta: Any,
    ) -> Self:
        assert identifier is None or len(identifier) > 0, "Identifier can't be empty"  # nosec: B101

        return cls(
            identifier=identifier or uuid4().hex,
            content=MultimodalContent.of(content),
            meta=meta,
        )

    identifier: str
    content: MultimodalContent
    meta: dict[str, Any]


@runtime_checkable
class ChoiceCompletion(Protocol):
    async def __call__(  # noqa: PLR0913
        self,
        *,
        instruction: Instruction | str,
        options: Sequence[ChoiceOption],
        input: Multimodal,  # noqa: A002
        prefill: str | None,
        toolbox: Toolbox,
        examples: Iterable[tuple[Multimodal, ChoiceOption]] | None,
        **extra: Any,
    ) -> ChoiceOption: ...


class SelectionException(Exception):
    pass
