from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any

from draive.choice.errors import SelectionException
from draive.choice.model import ChoiceOption
from draive.instructions import Instruction
from draive.lmm import Toolbox, lmm_invocation
from draive.scope import ctx
from draive.types import (
    LMMCompletion,
    LMMContextElement,
    LMMInput,
    LMMToolRequests,
    Multimodal,
    MultimodalContent,
    xml_tag,
)
from draive.types.lmm import LMMToolResponse

__all__ = [
    "lmm_choice_completion",
]


async def lmm_choice_completion(  # noqa: C901, PLR0912, PLR0913
    *,
    instruction: Instruction | str,
    options: Sequence[ChoiceOption],
    input: Multimodal,  # noqa: A002
    prefill: str | None,
    toolbox: Toolbox,
    examples: Iterable[tuple[Multimodal, ChoiceOption]] | None,
    **extra: Any,
) -> ChoiceOption:
    with ctx.nested(
        "lmm_choice_completion",
    ):
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

        if prefill := prefill:
            context.append(LMMCompletion.of(prefill))

        recursion_level: int = 0
        while recursion_level <= toolbox.recursion_limit:
            match await lmm_invocation(
                instruction=extended_instruction,
                context=context,
                tools=toolbox.available_tools(),
                tool_selection=toolbox.tool_selection(recursion_level=recursion_level),
                output="text",
                stream=False,
                **extra,
            ):
                case LMMCompletion() as completion:
                    ctx.log_debug("Received choice results")
                    if selection := xml_tag("CHOICE", source=completion.content):
                        if option := options_map.get(selection.as_string()):
                            return option

                    raise SelectionException("Invalid or missing selection")

                case LMMToolRequests() as tool_requests:
                    ctx.log_debug("Received choice tool calls")
                    responses: list[LMMToolResponse] = await toolbox.respond(tool_requests)

                    if direct_content := [
                        response.content for response in responses if response.direct
                    ]:
                        if selection := xml_tag(
                            "CHOICE",
                            source=MultimodalContent.of(*direct_content),
                        ):
                            if option := options_map.get(selection.as_string()):
                                return option

                        raise SelectionException("Invalid or missing selection")

                    elif prefill := prefill:  # move prefill to the next completion if used tools
                        del context[-1]
                        context.extend([tool_requests, *responses, LMMCompletion.of(prefill)])

                    else:
                        context.extend([tool_requests, *responses])

            recursion_level += 1  # continue with next recursion level

        raise RuntimeError("LMM exceeded limit of recursive calls")


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
            "</INPUT>",
            "<OPTIONS>",
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
Place identifier of the final choice inside a <CHOICE> XML tag within the result, \
like this: `<CHOICE>identifier</CHOICE>`.
</FORMAT>
"""
