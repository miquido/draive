from collections.abc import Iterable, Sequence
from typing import Any

from draive.lmm import AnyTool, Toolbox, lmm_invocation
from draive.parameters import DataModel
from draive.scope import ctx
from draive.types import (
    Instruction,
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMInstruction,
    LMMToolRequests,
    LMMToolResponse,
    MultimodalContent,
    MultimodalContentElement,
)

__all__: list[str] = [
    "lmm_generate_model",
]


async def lmm_generate_model[Generated: DataModel](  # noqa: PLR0913
    generated: type[Generated],
    /,
    *,
    instruction: Instruction | str,
    input: MultimodalContent | MultimodalContentElement,  # noqa: A002
    schema_variable: str | None = None,
    tools: Toolbox | Sequence[AnyTool] | None = None,
    examples: Iterable[tuple[MultimodalContent | MultimodalContentElement, Generated]]
    | None = None,
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

        instruction_message: LMMContextElement
        if variable := schema_variable:
            instruction_message = LMMInstruction.of(
                generation_instruction.updated(
                    **{variable: generated.json_schema()},
                ),
            )

        else:
            instruction_message = LMMInstruction.of(
                generation_instruction.extended(
                    DEFAULT_INSTRUCTION_EXTENSION,
                    joiner="\n\n",
                    schema=generated.json_schema(),
                )
            )

        context: list[LMMContextElement] = [
            instruction_message,
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

        for recursion_level in toolbox.call_range:
            match await lmm_invocation(
                context=context,
                tools=toolbox.available_tools(recursion_level=recursion_level),
                require_tool=toolbox.tool_suggestion(recursion_level=recursion_level),
                output="json",
                stream=False,
                **extra,
            ):
                case LMMCompletion() as completion:
                    return generated.from_json(completion.content.as_string())

                case LMMToolRequests() as tool_requests:
                    context.append(tool_requests)
                    responses: list[LMMToolResponse] = await toolbox.respond(tool_requests)

                    if direct_responses := [response for response in responses if response.direct]:
                        # TODO: check if this join makes any sense,
                        # perhaps we could merge json objects instead?
                        return generated.from_json(
                            "".join(
                                *[response.content.as_string() for response in direct_responses]
                            )
                        )

                    else:
                        context.extend(responses)

    # fail if we have not provided a result until this point
    raise RuntimeError("Failed to produce conversation completion")


DEFAULT_INSTRUCTION_EXTENSION: str = """\
IMPORTANT!
The result have to conform to the following JSON Schema:
```
{schema}
```
Provide ONLY a single, raw, valid JSON without any comments or formatting.
"""
