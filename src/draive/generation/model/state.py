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
        """Generate an instance of the requested state type.

        The type is always passed to the model as its requested output, letting
        each provider use its schema backed mode where the API offers one, fall
        back to a schema-less json mode where it does not, and finally to plain
        output where neither exists. ``schema_injection`` additionally delivers
        the schema through the instructions, which is what reaches providers
        without a schema backed mode of their own.

        Parameters
        ----------
        generated : type[Generated]
            Serializable state type to produce.
        instructions : Template | ModelInstructions, default=""
            Instructions passed to the model. A ``{model_schema}`` placeholder
            is replaced according to ``schema_injection``.
        input : Template | Multimodal
            Input content the generation is based on.
        schema_injection : Literal["full", "simplified", "skip"], default="skip"
            How to deliver the schema through the instructions - the complete
            json schema, a simplified rendering of it, or not at all. The
            placeholder is required for the schema to land; instructions
            without one are passed through unchanged.
        tools : Toolbox | Iterable[Tool], default=Toolbox.empty
            Tools available to the model while generating.
        examples : Iterable[tuple[Multimodal, Generated]], default=()
            Few-shot examples prepended to the generation context.
        decoder : ModelGenerationDecoder[Generated] | None, default=None
            Custom decoder reading the result back. When omitted, a json
            artifact is used when present, otherwise the completion text is
            decoded as json. It does not change what is requested from the
            model.
        **extra : Any
            Additional provider options forwarded to completion.

        Returns
        -------
        Generated
            Decoded instance of the requested type.
        """
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
