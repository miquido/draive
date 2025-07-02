from collections.abc import Iterable
from typing import Any, Literal, cast

from haiway import ctx

from draive.generation.model.types import ModelGeneratorDecoder
from draive.instructions import Instruction
from draive.lmm import (
    LMM,
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMInstruction,
    LMMToolRequests,
    LMMToolResponses,
)
from draive.multimodal import (
    MultimodalContent,
)
from draive.parameters import DataModel
from draive.prompts import Prompt
from draive.tools import Toolbox

__all__ = ("generate_model",)


async def generate_model[Generated: DataModel](  # noqa: C901, PLR0912
    generated: type[Generated],
    /,
    *,
    instruction: Instruction,
    input: Prompt | MultimodalContent,  # noqa: A002
    schema_injection: Literal["auto", "full", "simplified", "skip"],
    toolbox: Toolbox,
    examples: Iterable[tuple[MultimodalContent, Generated]] | None,
    decoder: ModelGeneratorDecoder | None,
    **extra: Any,
) -> Generated:
    with ctx.scope("generate_model"):
        extended_instruction: Instruction
        match schema_injection:
            case "auto":
                extended_instruction = instruction.extended(
                    DEFAULT_INSTRUCTION_EXTENSION.format(
                        schema=generated.simplified_schema(indent=2),
                    ),
                    joiner="\n\n",
                )

            case "full":
                extended_instruction = instruction.updated(
                    schema=generated.json_schema(indent=2),
                )

            case "simplified":
                extended_instruction = instruction.updated(
                    schema=generated.simplified_schema(indent=2),
                )

            case "skip":
                extended_instruction = instruction

        context: list[LMMContextElement] = [
            *[
                message
                for example in examples or []
                for message in [
                    LMMInput.of(example[0]),
                    LMMCompletion.of(example[1].to_json(indent=2)),
                ]
            ],
        ]

        match input:
            case Prompt() as prompt:
                context.extend(prompt.content)

            case value:
                context.append(LMMInput.of(value))

        formatted_instruction: LMMInstruction = Instruction.formatted(extended_instruction)
        repetition_level: int = 0
        while True:
            match await LMM.completion(
                instruction=formatted_instruction,
                context=context,
                tools=toolbox.available_tools(repetition_level=repetition_level),
                output=generated,  # provide model specification
                **extra,
            ):
                case LMMCompletion() as completion:
                    if decoder := decoder:
                        return generated.from_mapping(decoder(completion.content))

                    elif artifacts := completion.content.artifacts(generated):
                        return cast(Generated, artifacts[0])

                    else:
                        return generated.from_json(completion.content.to_str())

                case LMMToolRequests() as tool_requests:
                    context.append(tool_requests)
                    tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                    if direct_responses := [
                        response
                        for response in tool_responses.responses
                        if response.handling == "direct_result"
                    ]:
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
                            return generated.from_mapping(decoder(direct_responses_content))

                        else:
                            return generated.from_json(direct_responses_content.to_str())

                    else:
                        context.append(tool_responses)

            repetition_level += 1  # continue with next recursion level


DEFAULT_INSTRUCTION_EXTENSION: str = """\
<FORMAT>
Provide the result using a single raw valid JSON object that adheres strictly to the given \
SCHEMA without any comments, formatting, or additional elements.
<SCHEMA>
{schema}
</SCHEMA>
</FORMAT>
"""
