from collections.abc import Iterable, Sequence
from typing import Any, Literal

from draive.generation.model.generator import ModelGeneratorDecoder
from draive.lmm import AnyTool, Toolbox, lmm_invocation
from draive.parameters import DataModel
from draive.scope import ctx
from draive.types import (
    Instruction,
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequests,
    LMMToolResponse,
    MultimodalContent,
    MultimodalContentConvertible,
)

__all__: list[str] = [
    "lmm_generate_model",
]


async def lmm_generate_model[Generated: DataModel](  # noqa: PLR0913, C901, PLR0912
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
) -> Generated:
    with ctx.nested("lmm_generate_model"):
        toolbox: Toolbox
        match tools:
            case None:
                toolbox = Toolbox()

            case Toolbox() as tools:
                toolbox = tools

            case [*tools]:
                toolbox = Toolbox(*tools)

        generation_instruction: Instruction
        match instruction:
            case str(instruction):
                generation_instruction = Instruction(instruction)

            case Instruction() as instruction:
                generation_instruction = instruction

        extended_instruction: Instruction
        match schema_injection:
            case "auto":
                extended_instruction = Instruction.of(
                    generation_instruction.extended(
                        DEFAULT_INSTRUCTION_EXTENSION.format(
                            schema=generated.simplified_schema(indent=2),
                        ),
                        joiner="\n\n",
                    ),
                )

            case "full":
                extended_instruction = Instruction.of(
                    generation_instruction,
                    schema=generated.json_schema(indent=2),
                )

            case "simplified":
                extended_instruction = Instruction.of(
                    generation_instruction,
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
        while recursion_level <= toolbox.recursion_limit:
            match await lmm_invocation(
                instruction=extended_instruction,
                context=context,
                tools=toolbox.available_tools(),
                tool_requirement=toolbox.tool_requirement(recursion_level=recursion_level),
                output="json",
                stream=False,
                **extra,
            ):
                case LMMCompletion() as completion:
                    ctx.log_debug("Received model generation result")
                    if decoder := decoder:
                        return generated.from_dict(decoder(completion.content))

                    else:
                        return generated.from_json(completion.content.as_string())

                case LMMToolRequests() as tool_requests:
                    ctx.log_debug("Received model generation tool calls")
                    context.append(tool_requests)
                    responses: list[LMMToolResponse] = await toolbox.respond(tool_requests)

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
The output have to be a JSON conforming to the following schema:
```
{schema}
```
Provide ONLY a single, raw, valid JSON without any comments, formatting or additional elements.
"""
