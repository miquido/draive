from collections.abc import Iterable
from typing import Any, Literal, Protocol, runtime_checkable

from draive.instructions import Instruction
from draive.lmm import AnyTool, Toolbox
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel

__all__ = [
    "ModelGenerator",
    "ModelGeneratorDecoder",
]


@runtime_checkable
class ModelGeneratorDecoder(Protocol):
    def __call__(
        self,
        generated: MultimodalContent,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ModelGenerator(Protocol):
    async def __call__[Generated: DataModel](  # noqa: PLR0913
        self,
        generated: type[Generated],
        /,
        *,
        instruction: Instruction | str,
        input: Multimodal,  # noqa: A002
        schema_injection: Literal["auto", "full", "simplified", "skip"],
        tools: Toolbox | Iterable[AnyTool] | None,
        examples: Iterable[tuple[Multimodal, Generated]] | None,
        decoder: ModelGeneratorDecoder | None,
        **extra: Any,
    ) -> Generated: ...
