import json

from haiway import ctx, traced

from draive.instructions import Instruction, InstructionDeclaration
from draive.multimodal import MultimodalContent
from draive.multimodal.tags import MultimodalTagElement
from draive.parameters import DataModel
from draive.stages import Stage

__all__ = [
    "prepare_instruction",
]


@traced
async def prepare_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: InstructionDeclaration | str,
    /,
    *,
    guidelines: str | None = None,
) -> Instruction:
    ctx.log_info("Preparing instruction...")

    instruction_declaration: InstructionDeclaration
    match instruction:
        case str() as description:
            instruction_declaration = InstructionDeclaration(
                name="instruction",
                description=description,
                arguments=(),
            )

        case declaration:
            instruction_declaration = declaration

    result: MultimodalContent = await _preparation_stage(
        instruction=instruction_declaration,
        guidelines=guidelines,
    ).execute()

    ctx.log_info("...instruction preparation finished!")

    if parsed := MultimodalTagElement.parse_first("INSTRUCTION", content=result):
        return Instruction.of(
            parsed.content.as_string(),
            name=instruction_declaration.name,
            description=instruction_declaration.description,
        )

    else:
        raise ValueError("Failed to prepare instruction", result)


def _preparation_stage(
    *,
    instruction: InstructionDeclaration,
    guidelines: str | None = None,
) -> Stage:
    assert instruction.description is not None  # nosec: B101
    return Stage.completion(
        f"<DESCRIPTION>{instruction.description}</DESCRIPTION>"
        f"<ARGUMENTS>{_format_arguments(instruction)}<ARGUMENTS>"
        if instruction.arguments
        else f"<DESCRIPTION>{instruction.description}</DESCRIPTION>",
        instruction=PREPARE_INSTRUCTION.format(
            guidelines=f"<GUIDELINES>{guidelines}</GUIDELINES>" if guidelines else "",
        ),
    )


def _format_arguments(
    instruction: InstructionDeclaration,
) -> str:
    arguments: str = "<ARGUMENTS>"
    for argument in instruction.arguments:
        arguments += f"<{argument.name}>{json.dumps(argument.specification)}</{argument.name}>"
    arguments += "</ARGUMENTS>"
    return arguments


PREPARE_INSTRUCTION: str = """\
<INSTRUCTION>
You will be given a description of the instruction or task. Deeply analyze its meaning, context, variables and restrictions to prepare a detailed, followable instructions for completing the described task without any ambiguities.
</INSTRUCTION>

<STEPS>
Understand the subject
- Thoroughly review the instruction description, understand its purpose, structure, and key details
- Identify any nuances, requirements, and the intended outcomes
- Think step by step and breakdown its content to individual pieces for better understanding
- Examine provided context and variables

Describe in detail
- Prepare concise and highly precise instructions while retaining essential details
- Add necessary context or clarifications to improve usability and understanding
- Use placeholders for described variables (e.g., `{{variable}}`) using their names where appropriate

Format for readability
- Apply clear, logical formatting and structure to enhance readability and usability
</STEPS>
{guidelines}
<FORMAT>
Provide the final result containing only the improved instruction without any additional comments within a INSTRUCTION xml tag (e.g., `<INSTRUCTION>detailed instructions</INSTRUCTION>`).
</FORMAT>
"""  # noqa: E501
