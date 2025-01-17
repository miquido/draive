from asyncio import gather
from collections.abc import Callable, Coroutine, Mapping
from typing import Any

from haiway import MetricsHolder, as_list, ctx, traced

from draive.evaluation import (
    EvaluationSuite,
    EvaluationSuiteCase,
    SuiteEvaluatorResult,
)
from draive.generation import generate_text
from draive.instructions import Instruction, InstructionsRepository
from draive.multimodal.tags import MultimodalTagElement
from draive.parameters import DataModel

__all__ = [
    "optimize_instruction",
]


@traced
async def optimize_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: Instruction,
    /,
    *,
    guidelines: str | None = None,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    case_generator: Callable[
        [Instruction, SuiteEvaluatorResult[CaseParameters, Result]],
        Coroutine[None, None, EvaluationSuiteCase[CaseParameters]],
    ]
    | None = None,
    rounds_limit: int,
) -> Instruction:
    async with ctx.scope("optimize_instruction", metrics=MetricsHolder.handler()):
        assert rounds_limit > 0, "Rounds limit have to be geater than zero"  # nosec: B101
        ctx.log_info("Preparing instruction optimization...")

        instructions_repository: InstructionsRepository = ctx.state(InstructionsRepository)
        current_instruction: Instruction = instruction
        evaluation_cases: list[EvaluationSuiteCase[CaseParameters]] = as_list(
            await evaluation_suite.cases(reload=True)
        )

        for iteration in range(0, rounds_limit):
            ctx.log_info(f"...performing instruction optimization round ({iteration + 1})...")
            current_evaluation_result: SuiteEvaluatorResult[
                CaseParameters, Result
            ] = await _evaluate_instruction(
                current_instruction,
                instruction_repository=instructions_repository,
                evaluation_suite=evaluation_suite,
                evaluation_cases=evaluation_cases,
            )

            modified_instruction: Instruction = await _prepare_refined_instruction(
                current_instruction,
                evaluation_result=current_evaluation_result,
                guidelines=guidelines,
            )

            modified_evaluation_result: SuiteEvaluatorResult[
                CaseParameters, Result
            ] = await _evaluate_instruction(
                modified_instruction,
                instruction_repository=instructions_repository,
                evaluation_suite=evaluation_suite,
                evaluation_cases=evaluation_cases,
            )

            # if we have not improved try again starting from the same point
            if modified_evaluation_result.relative_score < current_evaluation_result.relative_score:
                ctx.log_info(
                    "...modified instruction worked worse, optimization round completed..."
                )
                continue  # use the previous instruction as this modification went worse

            ctx.log_info("...modified instruction worked better...")
            # update current instruction to better performing
            current_instruction = modified_instruction

            if not modified_evaluation_result.passed:
                ctx.log_info("...optimization round completed...")
                continue  # if we have not passed all cases keep improving

            ctx.log_info("...extending evaluation suite...")
            if case_generator is not None:
                evaluation_cases.append(
                    await case_generator(
                        modified_instruction,
                        modified_evaluation_result,
                    )
                )

            else:
                evaluation_cases.extend(
                    await evaluation_suite.generate_cases(
                        count=1,
                        guidelines=guidelines,
                    )
                )

            ctx.log_info("...optimization round completed...")

        ctx.log_info("...optimization completed!")
        return current_instruction


async def _evaluate_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: Instruction,
    /,
    *,
    instruction_repository: InstructionsRepository,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    evaluation_cases: list[EvaluationSuiteCase[CaseParameters]],
) -> SuiteEvaluatorResult[CaseParameters, Result]:
    ctx.log_info("...evaluating instruction...")
    evaluation_result: SuiteEvaluatorResult[CaseParameters, Result]

    async def instruction_fetch(
        name: str,
        /,
        *,
        arguments: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> Instruction | None:
        if name == instruction.name:
            return instruction.updated(**arguments if arguments is not None else {})

        else:  # preserve other prompts unchanged
            return await instruction_repository.fetch(
                name,
                arguments=arguments,
                **extra,
            )

    with ctx.updated(InstructionsRepository(fetch=instruction_fetch)):
        evaluation_result = SuiteEvaluatorResult(
            cases=await gather(
                *[evaluation_suite(case.parameters) for case in evaluation_cases],
                return_exceptions=False,
            )
        )

    ctx.log_debug(
        "...instruction perfomance: %.1f\nINSTRUCTION:\n%s\n---\nISSUES:\n%s",
        evaluation_result.relative_score * 100,
        instruction.content,
        evaluation_result.report(
            include_details=True,
            include_passed=False,
        ),
    )

    return evaluation_result


async def _prepare_refined_instruction[CaseParameters: DataModel, Result: DataModel | str](
    current_instruction: Instruction,
    /,
    *,
    guidelines: str | None,
    evaluation_result: SuiteEvaluatorResult[CaseParameters, Result],
) -> Instruction:
    ctx.log_info("...preparing instruction modification...")
    report: str = evaluation_result.report(
        include_passed=True,
        include_details=False,
    )
    analysis: str
    if result := MultimodalTagElement.parse_first(
        await generate_text(
            instruction=REVIEW_INSTRUCTION.format(
                guidelines=guidelines if guidelines is not None else "",
            ),
            input=f"\n<EVALUATION_RESULT>\n{report}\n</EVALUATION_RESULT>",
        ),
        tag="RESULT",
    ):
        analysis = result.content.as_string()

    else:
        ctx.log_error("...failed to generate instruction analysis...")
        raise RuntimeError()  # TODO: FIXME

    if result := MultimodalTagElement.parse_first(
        await generate_text(
            instruction=REFINE_INSTRUCTION.format(
                guidelines=guidelines if guidelines is not None else "",
            ),
            input=f"<SUBJECT>\n{current_instruction.content}\n</SUBJECT>\n"
            f"\n<ANALYSIS>\n{analysis}\n</ANALYSIS>",
            examples=REFINE_EXAMPLES,
        ),
        tag="RESULT",
    ):
        return Instruction(
            name=current_instruction.name,
            description=current_instruction.description,
            content=result.content.as_string(),
            arguments=current_instruction.arguments,
        )

    else:
        ctx.log_error("...failed to generate modified instruction...")
        raise RuntimeError()  # TODO: FIXME


# TODO: FIXME: refine instruction
REVIEW_INSTRUCTION: str = """\
<INSTRUCTION>
You will be given the report from the process evaluation suite run within the EALUATION_RESULT tag. Analyze its contents and prepare detailed conclusions which could serve as a foundation for improving the evaluated process.
</INSTRUCTION>

<STEPS>
Understand the Evaluation Results
- Carefully review the evaluation suite results to understand their structure and content.
- Identify the key areas being evaluated, such as clarity, usability, precision, or completeness.

Identify Weaknesses
- Highlight areas where the process failed to meet expectations or received negative feedback.
- Look for patterns in the results, such as repeated issues or recurring feedback themes.

Pinpoint Strengths
- Identify areas where the process performed well or received positive feedback.
- Determine what aspects contributed to their success (e.g., concise phrasing, clear formatting).

Extract Notes
- Convert weaknesses into specific, actionable recommendations for improvement.
- Document strengths as guidelines to retain or replicate in future improvements.
- Ensure all notes are specific, measurable, and directly tied to the evaluation findings.

Organize and Prioritize
- Categorize notes based on themes such as clarity, structure, or user-friendliness.
- Prioritize recommendations according to their impact on improving the process.

Deliver Findings
- Summarize prepared notes clearly and concisely, grouping them by priority or category.
- Format the output in a structured manner, ensuring it can be easily referenced during refinement.
</STEPS>
{guidelines}
<FORMAT>
The final result containing only the analysis findings HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>analysis findings</RESULT>`.
</FORMAT>
"""  # noqa: E501
# TODO: FIXME: refine instruction
REFINE_INSTRUCTION: str = """\
<INSTRUCTION>
You will be given the instruction within SUBJECT tag and its performance nalysis within ANALYSIS tag. Analyze its contents and prepare refined SUBJECT instruction according to ANALYSIS recoommendations.
</INSTRUCTION>

<STEPS>
Understand the Original Instructions
- Thoroughly review the SUBJECT instructions, understand their purpose, structure, and key details.
- Identify any nuances, requirements, and the intended outcomes.
- Think step by step and breakdown its content to individual pieces for beetter understanding.

Analyze Feedback
- Examine evaluation results within ANALYSIS to pinpoint unclear areas, weaknesses, and opportunities for improvement.

Refine and Enhance
- Rewrite the SUBJECT instructions to be concise and highly precise while retaining essential details.
- Add necessary context or clarifications to improve usability and understanding.
- Preserve all placeholder variables (e.g., `{{variable}}`) exactly as they appear, ensuring their names remain descriptive of their intended content.

Format for Readability
- Apply clear, logical formatting and structure to enhance readability and usability.
- Put each key section of the insctucion in capitalized xml tag.

Validate Against Criteria
- Ensure the refined instructions meet all defined criteria and exceed the quality of the original instructions.
</STEPS>
{guidelines}
<FORMAT>
The final result containing only the improved instructions HAVE to be put inside a `RESULT`\
 xml tag within the result i.e. `<RESULT>improved instructions</RESULT>`.
</FORMAT>
"""  # noqa: E501

# TODO: FIXME: prepare examples
REFINE_EXAMPLES = ()
