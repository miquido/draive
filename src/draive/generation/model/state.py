from collections.abc import Iterable, Mapping
from typing import Any, Literal, overload

from haiway import State, ctx, statemethod

from draive.generation.model.default import generate_model
from draive.generation.model.types import ModelGenerating, ModelGenerationDecoder
from draive.models import ModelInstructions, Tool, Toolbox
from draive.multimodal import Multimodal, MultimodalContent, Template, TemplatesRepository
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
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
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
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
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
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,  # noqa: A002
        schema_injection: Literal["full", "simplified", "skip"] = "skip",
        tools: Toolbox | Iterable[Tool] = (),
        examples: Iterable[tuple[Multimodal, Generated]] = (),
        decoder: ModelGenerationDecoder[Generated] | None = None,
        **extra: Any,
    ) -> Generated:
        async with ctx.scope("generate_model"):
            ctx.record_info(
                attributes={
                    "generated.model": generated.__qualname__,
                    "generated.schema_injection": schema_injection,
                },
            )
            instruction_arguments: Mapping[str, Multimodal] | None
            match schema_injection:
                case "full":
                    instruction_arguments = {
                        "model_schema": generated.json_schema(indent=2),
                    }

                case "simplified":
                    instruction_arguments = {
                        "model_schema": generated.simplified_schema(indent=2),
                    }

                case "skip":  # instruction is not modified
                    instruction_arguments = None

            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )

            if isinstance(input, Template):
                ctx.record_info(
                    attributes={"input.template": input.identifier},
                )

            return await self.generating(
                generated,
                # resolve instructions templates
                instructions=await TemplatesRepository.resolve_str(
                    instructions,
                    arguments=instruction_arguments,
                ),
                # resolve input templates
                input=await TemplatesRepository.resolve(input),
                toolbox=Toolbox.of(tools),
                examples=((MultimodalContent.of(ex_in), ex_out) for ex_in, ex_out in examples),
                decoder=decoder,
                **extra,
            )

    generating: ModelGenerating = generate_model
