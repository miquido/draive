from collections.abc import Iterable, Mapping
from typing import Any, Literal, overload

from haiway import State, ctx, statemethod

from draive.generation.model.default import generate_model
from draive.generation.model.types import ModelGenerating, ModelGenerationDecoder
from draive.models import ModelInstructions
from draive.multimodal import Multimodal, Template, TemplatesRepository
from draive.tools import Tool, Toolbox
from draive.utils.schema import simplified_schema

__all__ = ("ModelGeneration",)


class ModelGeneration(State):
    @overload
    @classmethod
    async def generate[Generated: State](
        cls,
        generated: type[Generated],
        /,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        schema_injection: Literal["full", "simplified", "skip"] = "skip",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        examples: Iterable[tuple[Multimodal, Generated]] = (),
        decoder: ModelGenerationDecoder[Generated] | None = None,
        **extra: Any,
    ) -> Generated: ...

    @overload
    async def generate[Generated: State](
        self,
        generated: type[Generated],
        /,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,
        schema_injection: Literal["full", "simplified", "skip"] = "skip",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        examples: Iterable[tuple[Multimodal, Generated]] = (),
        decoder: ModelGenerationDecoder[Generated] | None = None,
        **extra: Any,
    ) -> Generated: ...

    @statemethod
    async def generate[Generated: State](
        self,
        generated: type[Generated],
        /,
        *,
        instructions: Template | ModelInstructions = "",
        input: Template | Multimodal,  # noqa: A002
        schema_injection: Literal["full", "simplified", "skip"] = "skip",
        tools: Toolbox | Iterable[Tool] = Toolbox.empty,
        examples: Iterable[tuple[Multimodal, Generated]] = (),
        decoder: ModelGenerationDecoder[Generated] | None = None,
        **extra: Any,
    ) -> Generated:
        async with ctx.scope("model_generation"):
            assert generated.__SERIALIZABLE__  # nosec: B101
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
                        "model_schema": simplified_schema(
                            generated.__SPECIFICATION__,
                            indent=2,
                        ),
                    }

                case "skip":  # instruction is not modified
                    instruction_arguments = None

            if isinstance(instructions, Template):
                ctx.record_info(
                    attributes={"instructions.template": instructions.identifier},
                )
                instructions = await TemplatesRepository.resolve_str(
                    instructions,
                    arguments=instruction_arguments,
                )

            elif instruction_arguments:
                instructions = instructions.format_map(instruction_arguments)

            if isinstance(input, Template):
                ctx.record_info(
                    attributes={"input.template": input.identifier},
                )
                input = await TemplatesRepository.resolve(input)  # noqa: A001

            return await self._generating(
                generated,
                instructions=instructions,
                input=input,
                toolbox=Toolbox.of(tools),
                examples=examples,
                decoder=decoder,
                **extra,
            )

    _generating: ModelGenerating = generate_model

    def __init__(
        self,
        generating: ModelGenerating = generate_model,
    ) -> None:
        super().__init__(_generating=generating)
