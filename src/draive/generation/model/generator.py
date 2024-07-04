from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from draive.lmm import AnyTool, Toolbox
from draive.parameters import DataModel
from draive.types import Instruction, MultimodalContent, MultimodalContentConvertible

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
        input: MultimodalContent | MultimodalContentConvertible,  # noqa: A002
        schema_injection: Literal["auto", "full", "simplified", "skip"] = "auto",
        tools: Toolbox | Sequence[AnyTool] | None = None,
        examples: Iterable[tuple[MultimodalContent | MultimodalContentConvertible, Generated]]
        | None = None,
        decoder: ModelGeneratorDecoder | None = None,
        **extra: Any,
    ) -> Generated: ...
