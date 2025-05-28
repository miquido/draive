import json

from haiway import ctx

from draive.instructions import Instruction, InstructionDeclaration
from draive.multimodal import MultimodalContent
from draive.multimodal.tags import MultimodalTagElement
from draive.parameters import DataModel
from draive.stages import Stage

__all__ = [
    "InstructionPreparationAmbiguity",
    "prepare_instruction",
]


class InstructionPreparationAmbiguity(Exception):
    def __init__(
        self,
        *args: object,
        questions: str,
    ) -> None:
        super().__init__(*args)
        self.questions: str = questions


async def prepare_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: InstructionDeclaration | str,
    /,
    *,
    guidelines: str | None = None,
) -> Instruction:
    with ctx.scope("prepare_instruction"):
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

        assert instruction_declaration.description is not None  # nosec: B101
        result: MultimodalContent = await Stage.completion(
            f"<USER_TASK>{instruction_declaration.description}{_format_variables(instruction_declaration)}</USER_TASK>",
            instruction=PREPARE_INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>{guidelines}</GUIDELINES>\n" if guidelines else "",
            ),
        ).execute()

        if parsed := MultimodalTagElement.parse_first("RESULT_INSTRUCTION", content=result):
            ctx.log_info("...instruction preparation finished!")
            return Instruction.of(
                parsed.content.to_str(),
                name=instruction_declaration.name,
                description=instruction_declaration.description,
            )

        elif parsed := MultimodalTagElement.parse_first("QUESTIONS", content=result):
            ctx.log_error("...instruction preparation requires clarification!")
            raise InstructionPreparationAmbiguity(questions=parsed.content.to_str())

        else:
            ctx.log_error("...instruction preparation failed!")
            raise ValueError("Failed to prepare instruction", result)


def _format_variables(
    instruction: InstructionDeclaration,
) -> str:
    if not instruction.arguments:
        return "<VARIABLES>N/A</VARIABLES>"

    arguments: str = ""
    for argument in instruction.arguments:
        arguments += f"<{argument.name}>{json.dumps(argument.specification)}</{argument.name}>"

    return f"<VARIABLES>{arguments}</VARIABLES>"


PREPARE_INSTRUCTION: str = """\
You are an expert prompt engineer preparing system instructions for LLMs. Your goal is to create detailed, actionable instructions for completing the described task without any ambiguities.
You should maintain a clear and concise style, ensuring that the instructions are easy to understand and follow.

<INSTRUCTIONS>
You will be given a description of a task. Thoroughly analyze the meaning, context, variables and restrictions to prepare a detailed, actionable instructions required to complete the task.
When the task is ambiguous or lacks important information provide complete set of questions required to clarify.
When the task is not possible to describe or not achievable say "Task not achievable".
</INSTRUCTIONS>

<STEPS>
Understand the subject
- thoroughly review the task description, understand its purpose, structure, and key details
- identify any nuances, requirements, and the intended outcomes
- break down the task into individual pieces for better understanding
- examine provided context and variables

Describe in detail
- prepare concise and highly precise instructions while retaining essential details
- use placeholders for available variables where their content is appropriate refering to them by their names (e.g., `{{variable}}`)
- do not add any variables when there are none available
- use sections below and incorporate any relevant elements into the instruction for structural soundness
- rephrase section titles to fit the context of the instruction

<SECTIONS>
Introduction and role
- provide the context about the role which should be taken on or what goals and overarching tasks you want it to undertake
- it's best to put context early in the body of the instruction
- this section is suitable for most instructions

Tone guidance
- if the tone or manner of delivery significantly affects results quality, include guidance on tone
- this element may not be necessary depending on the task

Detailed task description and rules
- expand on the specific task you want to be done, as well as any rules that must be followed
- this is also where you can give instructions on handling irrelevant, ambiguous or impossible to handle inputs
- this section is mandatory as it contains the actual task description

Examples
- provide an example of an ideal response, encase this in XML tags, feel free to provide multiple examples
- if you do provide multiple examples, include the context about what it is an example of, and enclose each example in its own set of XML tags
- make sure to provide examples of common edge cases
- use examples as disambiguation tool, skip it for trivial tasks

Input data to process
- if there is data that needs to be processed within the instruction, include it here within relevant XML tags
- include multiple pieces of data when needed, but be sure to enclose each in its own set of XML tags
- this section may not be necessary depending on the task

Additional guidance, process description and rules
- for tasks with multiple steps or complex logic, it's good to describe detailed subtasks or encourage to analyze and think before giving an answer
- sometimes, you might have to even state "Before you give your answer..." just to make sure it is done first
- not necessary with all instructions, though if included, it's best to do this toward the end or right before the formatting section

Output formatting
- if there is a specific way you need the response formatted, clearly describe what that format is
- this element may not be necessary when the task does not require any specific format
- if you include it, putting it toward the end of the prompt is better than at the beginning
- encourage putting results in dedicated XML tags when more content is expected to be produced
</SECTIONS>

Format for clarity
- apply clear, logical formatting and structure to enhance readability and usability
- add necessary context or clarifications to improve usability and understanding
- use xml tags (including nested) with snake case names to wrap sections and separate important parts of the instruction
- always use English language for the instructions description

</STEPS>
{guidelines}
<EXAMPLE>
user: <USER_TASK>Summarize the content using only sentence equivalents. Make it short.<VARIABLES>N/A</VARIABLES></USER_TASK>
assistant: User task is to summarize some the provided content. There are two main restrictions:
- using only sentence equivalents
- providing short response
I need to prepare a detailed instruction based on that requirements.

<RESULT_INSTRUCTION>
You are a summarization bot preparing short summaries of provided content.
Use the same tone and style as the original.

<INSTRUCTIONS>
You will be given a content that needs to be summarized including all relevant details.
When the content is empty respond with "N/A".
</INSTRUCTIONS>

<GUIDELINES>
Use only sentence equivalents within the result. Summary should be as concise and short as possible without giving up any important details. Make sure that all factual and logical elements stay the same as in the source. If there are any contradictions or inconsistencies in the source, make sure to contain them in the summary.
</GUIDELINES>
</RESULT_INSTRUCTION>
</EXAMPLE>

<FORMAT>
Provide the final result containing only the improved instruction without any additional comments within a RESULT_INSTRUCTION xml tag (e.g., `<RESULT_INSTRUCTION>detailed instructions</RESULT_INSTRUCTION>`).
When disambiguation questions are required provide the result containing only the questions within QUESTIONS xml tag (e.g., `<QUESTIONS>clarification questions</QUESTIONS>`) instead.
</FORMAT>
"""  # noqa: E501
