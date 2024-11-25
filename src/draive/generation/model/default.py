from collections.abc import Iterable
from typing import Any, Literal

from haiway import ctx

from draive.generation.model.types import ModelGeneratorDecoder
from draive.instructions import Instruction
from draive.lmm import (
    AnyTool,
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequests,
    LMMToolResponse,
    Toolbox,
    lmm_invoke,
)
from draive.multimodal import (
    Multimodal,
    MultimodalContent,
)
from draive.parameters import DataModel

__all__: list[str] = [
    "default_generate_model",
]


async def default_generate_model[Generated: DataModel](  # noqa: PLR0913, C901, PLR0912
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
) -> Generated:
    with ctx.scope("generate_model"):
        toolbox: Toolbox = Toolbox.of(tools)

        generation_instruction: Instruction
        match instruction:
            case str(instruction):
                generation_instruction = Instruction.of(instruction)

            case Instruction() as instruction:
                generation_instruction = instruction

        extended_instruction: Instruction
        match schema_injection:
            case "auto":
                extended_instruction = generation_instruction.extended(
                    DEFAULT_INSTRUCTION_EXTENSION.format(
                        schema=generated.simplified_schema(indent=2),
                    ),
                    joiner="\n\n",
                )

            case "full":
                extended_instruction = generation_instruction.updated(
                    schema=generated.json_schema(indent=2),
                )

            case "simplified":
                extended_instruction = generation_instruction.updated(
                    schema=generated.simplified_schema(indent=2),
                )

            case "skip":
                extended_instruction = Instruction.of(
                    generation_instruction,
                )

        context: list[LMMContextElement] = [
            *[
                message
                for example in examples or []
                for message in [
                    LMMInput.of(example[0]),
                    LMMCompletion.of(example[1].as_json(indent=2)),
                ]
            ],
            LMMInput.of(input),
        ]

        recursion_level: int = 0
        while recursion_level <= toolbox.repeated_calls_limit:
            match await lmm_invoke(
                instruction=extended_instruction,
                context=context,
                tool_selection=toolbox.tool_selection(repetition_level=recursion_level),
                tools=toolbox.available_tools(),
                output=generated,  # provide model specification
                **extra,
            ):
                case LMMCompletion() as completion:
                    if decoder := decoder:
                        return generated.from_dict(decoder(completion.content))

                    else:
                        return generated.from_json(completion.content.as_string())

                case LMMToolRequests() as tool_requests:
                    context.append(tool_requests)
                    responses: list[LMMToolResponse] = await toolbox.respond_all(tool_requests)

                    if direct_responses := [response for response in responses if response.direct]:
                        for response in direct_responses:
                            if isinstance(response, generated):
                                # return first response matching requested model
                                return response

                            else:
                                continue

                        direct_responses_content: MultimodalContent = MultimodalContent.of(
                            *[response.content for response in direct_responses]
                        )

                        # TODO: check if this join makes any sense,
                        # perhaps we could merge json objects instead?
                        if decoder := decoder:
                            return generated.from_dict(decoder(direct_responses_content))

                        else:
                            return generated.from_json(direct_responses_content.as_string())

                    else:
                        context.extend(responses)

            recursion_level += 1  # continue with next recursion level

        raise RuntimeError("LMM exceeded limit of recursive calls")


DEFAULT_INSTRUCTION_EXTENSION: str = """\
<FORMAT>
Provide the result using a single raw valid JSON object that adheres strictly to the given \
SCHEMA without any comments, formatting, or additional elements.
<SCHEMA>
{schema}
</SCHEMA>
</FORMAT>
"""
