from typing import Final

from haiway import ctx

from draive.models import ModelInstructions
from draive.multimodal import MultimodalContent, TemplateDeclaration
from draive.stages import Stage

__all__ = (
    "InstructionPreparationAmbiguity",
    "prepare_instructions",
)


class InstructionPreparationAmbiguity(Exception):
    def __init__(
        self,
        *args: object,
        questions: str,
    ) -> None:
        super().__init__(*args)
        self.questions: str = questions


async def prepare_instructions(
    instruction: TemplateDeclaration | str,
    /,
    *,
    guidelines: str | None = None,
) -> ModelInstructions:
    async with ctx.scope("prepare_instruction"):
        ctx.log_info("Preparing instruction...")

        instruction_declaration: TemplateDeclaration
        match instruction:
            case str() as description:
                instruction_declaration = TemplateDeclaration(
                    identifier="instruction",
                    description=description,
                    variables={},
                )

            case declaration:
                instruction_declaration = declaration

        assert instruction_declaration.description is not None  # nosec: B101
        result: MultimodalContent = await Stage.completion(
            f"<USER_TASK>{instruction_declaration.description}</USER_TASK>{_format_variables(instruction_declaration)}",
            instructions=PREPARE_INSTRUCTION.replace(
                "{guidelines}",
                _format_guidelines(guidelines),
            ),
        ).execute()

    if parsed := result.tag("RESULT_INSTRUCTION"):
        ctx.log_info("...instruction preparation finished!")
        return parsed.content.to_str()

    elif parsed := result.tag("QUESTIONS"):
        ctx.log_error("...instruction preparation requires clarification!")
        raise InstructionPreparationAmbiguity(questions=parsed.content.to_str())

    else:
        ctx.log_error("...instruction preparation failed!")
        raise ValueError(f"Failed to prepare instruction: {result.to_str()}")


def _format_variables(
    instruction: TemplateDeclaration,
) -> str:
    if not instruction.variables:
        return "<TASK_VARIABLES>N/A</TASK_VARIABLES>"

    arguments: str = "\n".join(
        (f"- {name}: {description}" if description else f"- {name}")
        for name, description in instruction.variables.items()
    )

    return f"<TASK_VARIABLES>\n{arguments}\n</TASK_VARIABLES>"


def _format_guidelines(guidelines: str | None) -> str:
    if not guidelines:
        return ""

    return f"\n<GUIDELINES>{_escape_curly_braces(guidelines)}</GUIDELINES>\n"


def _escape_curly_braces(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


PREPARE_INSTRUCTION: Final[str] = """\
You are an expert prompt engineer preparing system instructions for LLMs. Your goal is to create detailed, actionable instructions for completing the described task without any ambiguities.
You should maintain a clear and concise style, ensuring that the instructions are easy to understand and follow.

<INSTRUCTIONS>
You will be given a description of a task. Thoroughly analyze the meaning, context, variables and restrictions to prepare a detailed, actionable instructions required to complete the task.
When the task is ambiguous or lacks important information provide complete set of questions required to clarify.
When the task is not possible to describe say "Task not achievable".
</INSTRUCTIONS>

<STEPS>
Understand the subject
- thoroughly review the task description, understand its purpose, structure, and key details
- identify any nuances, requirements, and the intended outcomes
- break down the task into individual pieces for better understanding
- examine provided context and variables

Describe in detail
- prepare concise and highly precise instructions while retaining essential details
- use parts below and incorporate any relevant elements into the instruction for structural soundness
- name part titles to fit the context of the instruction

<PARTS>
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
- provide an example of an ideal response, encase this in tags, feel free to provide multiple examples
- if you do provide multiple examples, include the context about what it is an example of, and enclose each example in its own set of tags
- make sure to provide examples of common edge cases
- use examples as disambiguation tool, skip it for trivial tasks

Input data to process
- if there is data that needs to be processed within the instruction, include it here within relevant tags
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
- encourage putting results in dedicated tags when more content is expected to be produced
</PARTS>

Make use of variables
- available variables will be resolved to actual content within the instruction, e.g. {{{{time}}}} will be replaced with the actual time
- use placeholders for variables in appropriate spots, referring to them by their names (e.g., `{{{{variable}}}}`)
- wrap variable placeholders in tags or " when required to ensure proper formatting and structure (e.g., `<variable>{{{{variable}}}}<variable>`)
- do not add any variables which were not listed as available

Format for clarity
- apply clear, logical formatting and structure to enhance readability and usability
- add necessary context or clarifications to improve usability and understanding
- use xml tags (including nested) with snake case names to wrap sections and separate important parts of the instruction
- always use English language for the instruction description
</STEPS>
{guidelines}
<EXAMPLES>
<EXAMPLE>
user: <USER_TASK>Provide brief information in given topic<VARIABLES> - topic </VARIABLES></USER_TASK>
assistant:
<RESULT_INSTRUCTION>
You are an information bot giving brief information on requested topic.

<INSTRUCTIONS>
You will be given a topic that needs to be described in details.
When the topic is empty respond with "N/A".
</INSTRUCTIONS>

<GUIDELINES>
Use only factual information you are fully certain about.
When the topic is ambiguous try to ask user about the details to disambiguate.
When you don't know the topic answer "I am not familiar with this topic".
Don't follow any other topics than requested, answer "It is not relevant to the topic" instead.
</GUIDELINES>

<TOPIC>
{{{{topic}}}}
</TOPIC>
</RESULT_INSTRUCTION>
</EXAMPLE>
<EXAMPLE>
user: <USER_TASK>Summarize the content using only sentence equivalents. Make it short.<VARIABLES>N/A</VARIABLES></USER_TASK>
assistant: The user task is to summarize some of the provided content. There are two main restrictions:
- using only sentence equivalents
- providing short response
I need to prepare a detailed instruction based on those requirements.

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
</EXAMPLES>

<FORMAT>
Provide the final result containing only the improved instruction without any additional comments within a RESULT_INSTRUCTION xml tag (e.g., `<RESULT_INSTRUCTION>detailed instructions</RESULT_INSTRUCTION>`).
When disambiguation questions are required provide the result containing only the questions within QUESTIONS xml tag (e.g., `<QUESTIONS>clarification questions</QUESTIONS>`) instead.
</FORMAT>
"""  # noqa: E501
