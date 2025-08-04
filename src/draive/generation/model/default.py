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


async def generate_model[Generated: DataModel](  # noqa: C901, PLR0912, PLR0915
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
    async with ctx.scope("generate_model"):
        extended_instruction: Instruction
        match schema_injection:
            case "auto":
                extended_instruction = instruction.extended(
                    DEFAULT_INSTRUCTION_EXTENSION,
                    joiner="\n\n",
                    schema=generated.simplified_schema(indent=2),
                )

            case "full":
                extended_instruction = instruction.with_arguments(
                    schema=generated.json_schema(indent=2),
                )

            case "simplified":
                extended_instruction = instruction.with_arguments(
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

        formatted_instruction: LMMInstruction = extended_instruction.format()
        tools_turn: int = 0
        result: MultimodalContent = MultimodalContent.empty
        result_extension: MultimodalContent = MultimodalContent.empty
        while True:
            ctx.log_debug("...requesting completion...")
            match await LMM.completion(
                instruction=formatted_instruction,
                context=context,
                tools=toolbox.available_tools(tools_turn=tools_turn),
                output=generated,  # provide model specification
                **extra,
            ):
                case LMMCompletion() as completion:
                    ctx.log_debug("...received result...")
                    result = result_extension.appending(completion.content)
                    break  # proceed to resolving

                case LMMToolRequests() as tool_requests:
                    ctx.log_debug(f"...received tool requests (turn {tools_turn})...")
                    # skip tool_requests.content - no need for extra comments
                    tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                    if completion := tool_responses.completion(extension=result_extension):
                        ctx.log_debug("...received tools direct result...")
                        result = completion.content
                        break  # proceed to resolving

                    elif extension := tool_responses.completion_extension():
                        ctx.log_debug("...received tools result extension...")
                        result_extension = result_extension.appending(extension)

                    ctx.log_debug("...received tools responses...")
                    context.extend((tool_requests, tool_responses))

            tools_turn += 1  # continue with next turn

        try:
            ctx.log_debug("...decoding result...")
            if decoder := decoder:
                ctx.log_debug("...decoded with custom decoder!")
                return decoder(result)

            elif artifacts := result.artifacts(generated):
                ctx.log_debug("...direct artifact found!")
                return cast(Generated, artifacts[0])

            else:
                ctx.log_debug("...decoded from json string!")
                return generated.from_json(result.to_str())

        except Exception as exc:
            ctx.log_error(
                f"Failed to decode {generated.__name__} model due to an error: {type(exc)}",
                exception=exc,
            )
            raise exc


DEFAULT_INSTRUCTION_EXTENSION: str = """\
<FORMAT>
Provide the result using a single raw valid JSON object that adheres strictly to the given \
SCHEMA without any comments, formatting, or additional elements.
<SCHEMA>
{schema}
</SCHEMA>
</FORMAT>
"""
