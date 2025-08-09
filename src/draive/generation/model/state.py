from collections.abc import Iterable
from typing import Any, Literal, overload

from haiway import State, statemethod

from draive.generation.model.default import generate_model
from draive.generation.model.types import ModelGenerating, ModelGenerationDecoder
from draive.models import ResolveableInstructions, Tool, Toolbox
from draive.multimodal import Multimodal, MultimodalContent
from draive.parameters import DataModel

__all__ = ("ModelGeneration",)


class ModelGeneration(State):
    @overload
    @classmethod
    async def generate[Generated: DataModel](
        cls,
        generated: type[Generated],
        /,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,
        schema_injection: Literal["full", "simplified", "skip"] = "skip",
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, Generated]] = (),
        decoder: ModelGenerationDecoder[Generated] | None = None,
        **extra: Any,
    ) -> Generated: ...

    @overload
    async def generate[Generated: DataModel](
        self,
        generated: type[Generated],
        /,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,
        schema_injection: Literal["full", "simplified", "skip"] = "skip",
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, Generated]] = (),
        decoder: ModelGenerationDecoder[Generated] | None = None,
        **extra: Any,
    ) -> Generated: ...

    @statemethod
    async def generate[Generated: DataModel](
        self,
        generated: type[Generated],
        /,
        *,
        instructions: ResolveableInstructions = "",
        input: Multimodal,  # noqa: A002
        schema_injection: Literal["full", "simplified", "skip"] = "skip",
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, Generated]] = (),
        decoder: ModelGenerationDecoder[Generated] | None = None,
        **extra: Any,
    ) -> Generated:
        return await self.generating(
            generated,
            instructions=instructions,
            schema_injection=schema_injection,
            input=MultimodalContent.of(input),
            toolbox=Toolbox.of(tools),
            examples=((MultimodalContent.of(ex_in), ex_out) for ex_in, ex_out in examples),
            decoder=decoder,
            **extra,
        )

    generating: ModelGenerating = generate_model
