from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any

from haiway import ctx

from draive.choice.types import ChoiceOption, SelectionException
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
from draive.multimodal import Multimodal, MultimodalContent, MultimodalTagElement
from draive.tools import Toolbox

__all__ = ("default_choice_completion",)


async def default_choice_completion(  # noqa: C901
    *,
    instruction: Instruction | str,
    options: Sequence[ChoiceOption],
    input: Multimodal,  # noqa: A002
    toolbox: Toolbox,
    examples: Iterable[tuple[Multimodal, ChoiceOption]] | None,
    **extra: Any,
) -> ChoiceOption:
    with ctx.scope("choice_completion"):
        assert options, "Choice options cannot be empty"  # nosec: B101
        assert all(  # nosec: B101
            example[1] in options for example in examples or []
        ), "Choice options examples have to be included in available options"

        if len(options) == 1:
            return options[0]  # if there is ony one option it has to be the result

        options_map: dict[str, ChoiceOption] = {option.identifier: option for option in options}
        assert len(options_map) == len(options), "Choice options cannot have duplicate identifiers"  # nosec: B101

        extended_instruction: Instruction
        match instruction:
            case str() as instruction_string:
                extended_instruction = Instruction.of(
                    f"{instruction_string}\n\n{INSTRUCTION_EXTENSION}"
                )

            case instruction:
                extended_instruction = instruction.extended(
                    INSTRUCTION_EXTENSION,
                    joiner="\n\n",
                )

        formatted_options: MultimodalContent = MultimodalContent.of(
            *(_format_option(option) for option in options)
        )

        context: list[LMMContextElement] = [
            *chain.from_iterable(
                _format_example(
                    example,
                    options=formatted_options,
                )
                for example in examples or []
            ),
            _format_input(
                input,
                options=formatted_options,
            ),
        ]

        formatted_instruction: LMMInstruction = Instruction.formatted(extended_instruction)
        repetition_level: int = 0
        while True:
            match await LMM.completion(
                instruction=formatted_instruction,
                context=context,
                tools=toolbox.available_tools(repetition_level=repetition_level),
                output="text",
                **extra,
            ):
                case LMMCompletion() as completion:
                    ctx.log_debug("Received choice results")
                    if selection := MultimodalTagElement.parse_first(
                        "CHOICE",
                        content=completion.content,
                    ):
                        if option := options_map.get(selection.content.to_str()):
                            return option

                    raise SelectionException("Invalid or missing selection")

                case LMMToolRequests() as tool_requests:
                    ctx.log_debug("Received choice tool calls")
                    tool_responses: LMMToolResponses = await toolbox.respond_all(tool_requests)

                    if direct_content := [
                        response.content
                        for response in tool_responses.responses
                        if response.handling == "direct_result"
                    ]:
                        if selection := MultimodalTagElement.parse_first(
                            "CHOICE",
                            content=MultimodalContent.of(*direct_content),
                        ):
                            if option := options_map.get(selection.content.to_str()):
                                return option

                        raise SelectionException("Invalid or missing selection")

                    else:
                        context.extend(
                            [
                                tool_requests,
                                tool_responses,
                            ]
                        )

            repetition_level += 1  # continue with next recursion level


def _format_option(
    option: ChoiceOption,
    /,
) -> MultimodalContent:
    return MultimodalContent.of(
        f"<OPTION identifier='{option.identifier}'>",
        option.content,
        "</OPTION>",
    )


def _format_input(
    input: Multimodal,  # noqa: A002
    /,
    options: MultimodalContent,
) -> LMMInput:
    return LMMInput.of(
        MultimodalContent.of(
            "<INPUT>",
            input,
            "</INPUT>\n\n<OPTIONS>",
            options,
            "</OPTIONS>",
        )
    )


def _format_example(
    example: tuple[Multimodal, ChoiceOption],
    /,
    options: MultimodalContent,
) -> tuple[LMMInput, LMMCompletion]:
    return (
        _format_input(example[0], options=options),
        LMMCompletion.of(f"```\n{example[1].identifier}\n```"),
    )


INSTRUCTION_EXTENSION: str = """\
<FORMAT>
Ensure the result is following the strict format:
Place only the identifier of the chosen option inside a <CHOICE> tag, as shown here:

<CHOICE>option_identifier</CHOICE>

</FORMAT>
"""
