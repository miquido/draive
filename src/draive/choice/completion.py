from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from draive.choice.model import ChoiceOption
from draive.instructions import Instruction
from draive.lmm import Toolbox
from draive.types import Multimodal

__all__ = [
    "ChoiceCompletion",
]


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
