from collections.abc import Mapping
from typing import Any, Literal, overload

from haiway import State, ctx, traced

from draive.evaluation import (
    EvaluationSuite,
    SuiteEvaluatorResult,
)
from draive.instructions import Instruction, InstructionsRepository
from draive.lmm import LMMContext
from draive.lmm.types import LMMCompletion, LMMInput
from draive.multimodal import MultimodalContent
from draive.multimodal.tags import MultimodalTagElement
from draive.parameters import DataModel
from draive.utils import Processing
from draive.workflow import Stage, workflow_completion
from draive.workflow.types import StageCondition

__all__ = [
    "refine_instruction",
]


@overload
async def refine_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: Instruction,
    /,
    *,
    guidelines: str | None = None,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    rounds_limit: int,
) -> Instruction: ...


@overload
async def refine_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: Instruction,
    /,
    *,
    guidelines: str | None = None,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    from_scratch: Literal[True],
) -> Instruction: ...


@overload
async def refine_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: Instruction,
    /,
    *,
    guidelines: str | None = None,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    rounds_limit: int,
    from_scratch: bool,
) -> Instruction: ...


@traced
async def refine_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: Instruction,
    /,
    *,
    guidelines: str | None = None,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    rounds_limit: int = 0,
    from_scratch: bool = False,
) -> Instruction:
    assert rounds_limit > 0 or from_scratch  # nosec: B101
    ctx.log_info("Preparing instruction refinement...")

    result: MultimodalContent = await workflow_completion(
        _initialization_stage(
            instruction=instruction,
            guidelines=guidelines,
            from_scratch=from_scratch,
        ),
        _evaluation_stage(evaluation_suite=evaluation_suite),
        _refinement_loop_stage(
            evaluation_suite=evaluation_suite,
            guidelines=guidelines,
            rounds_limit=rounds_limit,
        ),
    )
    ctx.log_info("...instruction refinement finished!")

    return instruction.updated(content=result.as_string())


def _instructions_repository(
    *,
    instruction: Instruction,
) -> InstructionsRepository:
    instructions_repository: InstructionsRepository = ctx.state(InstructionsRepository)

    async def instruction_fetch(
        name: str,
        /,
        *,
        arguments: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> Instruction | None:
        if name == instruction.name:
            return instruction.with_arguments(**arguments if arguments is not None else {})

        else:  # preserve other prompts unchanged
            return await instructions_repository.fetch(
                name,
                arguments=arguments,
                **extra,
            )

    return InstructionsRepository(fetch=instruction_fetch)


class _RefinementState(State):
    instruction: Instruction
    evaluation: SuiteEvaluatorResult[Any, Any] | None
    round: int


def _initialization_stage(
    *,
    instruction: Instruction,
    guidelines: str | None,
    from_scratch: bool,
) -> Stage:
    if from_scratch:

        async def initialization(
            content: MultimodalContent,
        ) -> MultimodalContent:
            ctx.log_info("...initializing refinement...")
            # update initial instruction based on current result - prepared instruction
            prepared_instruction: Instruction = instruction.updated(content=content.as_string())
            # and prepare initial processing state
            await Processing.write(
                _RefinementState(
                    instruction=prepared_instruction,
                    evaluation=None,
                    round=0,
                )
            )

            return content  # no changes to the result

        return Stage.sequence(
            # prepare initial instruction
            Stage.completion(
                # should we include list of arguments? there will be no explanation though
                f"<SUBJECT>{instruction.content}</SUBJECT>"
                + (
                    f"\n<DESCRIPTION>{instruction.description}</DESCRIPTION>"
                    if instruction.description
                    else ""
                ),
                instruction=PREPARE_INSTRUCTION.format(
                    guidelines=f"<GUIDELINES>{guidelines}</GUIDELINES>" if guidelines else "",
                ),
            ),
            # initialize workflow state
            Stage.result_processing(initialization),
            # cut out preparation stage context - keep only the result
            Stage.context_trimming(limit=0),
        )

    else:

        async def initialization(
            content: MultimodalContent,
        ) -> MultimodalContent:
            ctx.log_info("...initializing refinement...")
            # prepare initial processing state
            await Processing.write(
                _RefinementState(
                    instruction=instruction,
                    evaluation=None,
                    round=0,
                )
            )

            # set the result to be initial instruction
            return MultimodalContent.of(instruction.content)

        return Stage.result_processing(initialization)


def _evaluation_stage[CaseParameters: DataModel, Result: DataModel | str](
    *,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
) -> Stage:
    async def evaluation(
        context: LMMContext,
        result: MultimodalContent,
    ) -> tuple[LMMContext, MultimodalContent]:
        processing_state: _RefinementState = await Processing.read(
            _RefinementState,
            required=True,
        )
        current_instruction: Instruction = processing_state.instruction.updated(
            content=result.as_string()
        )

        ctx.log_info("...evaluating current instruction...")
        current_evaluation: SuiteEvaluatorResult[CaseParameters, Result]
        with ctx.updated(_instructions_repository(instruction=current_instruction)):
            current_evaluation = await evaluation_suite()

        previous_evaluation: SuiteEvaluatorResult[CaseParameters, Result] | None
        previous_evaluation = processing_state.evaluation
        if previous_evaluation is None:
            ctx.log_info("...no previous evaluation using current as a baseline...")
            # if there was no evaluation yet use the current one
            await Processing.write(
                processing_state.updated(
                    instruction=current_instruction,
                    evaluation=current_evaluation,
                )
            )
            return (context, result)

        if current_evaluation.relative_score <= previous_evaluation.relative_score:
            ctx.log_info("...evaluation score worse, rolling back to previous instruction...")
            # rollback to previous instruction - processing state holds previous one
            return (context, MultimodalContent.of(processing_state.instruction.content))

        ctx.log_info("...evaluation score better, proceeding...")
        # continue with improved state otherwise
        await Processing.write(
            processing_state.updated(
                instruction=current_instruction,
                evaluation=current_evaluation,
            )
        )
        return (context, result)

    return Stage(evaluation)


def _refinement_loop_stage[CaseParameters: DataModel, Result: DataModel | str](
    *,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    guidelines: str | None,
    rounds_limit: int,
) -> Stage:
    return Stage.loop(
        Stage.sequence(
            _refinement_stage(guidelines=guidelines),
            _evaluation_stage(evaluation_suite=evaluation_suite),
            # cut out stage context - keep only the result
            Stage.context_trimming(limit=0),
        ),
        condition=_refinement_loop_condition(rounds_limit=rounds_limit),
    )


def _refinement_stage(
    *,
    guidelines: str | None,
) -> Stage:
    async def _analysis(
        context: LMMContext,
        result: MultimodalContent,
    ) -> tuple[LMMContext, MultimodalContent]:
        ctx.log_info("...analyzing evaluation results...")
        processing_state: _RefinementState = await Processing.read(
            _RefinementState,
            required=True,
        )
        evaluation: SuiteEvaluatorResult[Any, Any] | None = processing_state.evaluation
        if evaluation is None or not evaluation.cases:
            ctx.log_info("...evaluation results unavailable, skipping analysis...")
            # nothing to analyze if no evaluation was done
            return (context, result)

        formatted_evaluation_results: MultimodalContent = MultimodalContent.of(
            "\n<EALUATION_RESULT>",
            evaluation.report(
                include_passed=False,
                include_details=False,
            ),
            "</EALUATION_RESULT>",
        )
        analysis_recommendations: MultimodalTagElement | None = MultimodalTagElement.parse_first(
            "RECOMMENDATIONS",
            content=await Stage.completion(
                formatted_evaluation_results,
                instruction=REVIEW_INSTRUCTION,
            ).execute(),
        )

        if analysis_recommendations is None:
            raise ValueError("Missing evaluation analysis recommendations")

        return (
            (
                *context,
                LMMInput.of(formatted_evaluation_results),
                LMMCompletion.of(
                    MultimodalContent.of(
                        f"<SUBJECT>{processing_state.instruction.content}</SUBJECT>",
                        (
                            f"\n<DESCRIPTION>{processing_state.instruction.description}</DESCRIPTION>"
                            if processing_state.instruction.description
                            else ""
                        ),
                        "\n<RECOMMENDATIONS>",
                        analysis_recommendations.content,
                        "</RECOMMENDATIONS>",
                    ),
                ),
            ),
            result,
        )

    return Stage.sequence(
        Stage(_analysis),
        Stage.loopback_completion(
            instruction=REFINE_INSTRUCTION.format(
                guidelines=guidelines,
            ),
        ),
    )


def _refinement_loop_condition[CaseParameters: DataModel, Result: DataModel | str](
    *,
    rounds_limit: int,
) -> StageCondition:
    async def condition(
        context: LMMContext,
        result: MultimodalContent,
    ) -> bool:
        processing_state: _RefinementState = await Processing.read(
            _RefinementState,
            default=_RefinementState(
                instruction=Instruction.of(result.as_string()),
                evaluation=None,
                round=0,
            ),
        )
        ctx.log_info(f"...ending refinement round {processing_state.round}...")
        if processing_state.round >= rounds_limit:
            ctx.log_info("...refinement round limit reached...")
            return False

        if (
            processing_state.evaluation is not None
            and processing_state.evaluation.relative_score >= 0.99  # noqa: PLR2004
        ):
            ctx.log_info("...refinement evaluation passed...")
            return False

        await Processing.write(processing_state.updated(round=processing_state.round + 1))
        return True

    return condition


# TODO: FIXME: prepare instruction
PREPARE_INSTRUCTION: str = """\

"""
# TODO: FIXME: refine instruction
REVIEW_INSTRUCTION: str = """\
<INSTRUCTION>
You will be given the report from the evaluation suite run within the EALUATION_RESULT tag. Analyze its contents and prepare detailed conclusions which could serve as a foundation for improving the evaluated process. Subject of evaluation as an instruction, make sure your recommendations address the instruction improvements.
</INSTRUCTION>

<STEPS>
Understand the Evaluation Results
- Carefully review the evaluation suite results to understand their structure and content
- Identify the key areas being evaluated and their meaning

Identify Weaknesses
- Highlight areas where the process failed to meet expectations or received negative feedback
- Look for patterns in the results, such as repeated issues or recurring feedback themes

Extract Notes
- Convert weaknesses into specific, actionable recommendations for improvement
- Ensure all notes are specific, measurable, and directly tied to the evaluation findings

Organize and Prioritize
- Categorize notes based on themes and areas of improvement
- Prioritize recommendations according to their impact on improving the process

Deliver Findings
- Summarize prepared notes clearly and concisely, in form of actionnable recommendations
- Format the output in a structured manner, ensuring it can be easily referenced during refinement
</STEPS>
{guidelines}
<FORMAT>
The final result containing only the improvement recommendations HAVE to be put inside a
 `RECOMMENDATIONS` xml tag within the result i.e. \
 `<RECOMMENDATIONS>analysis recommendations</RECOMMENDATIONS>`.
</FORMAT>
"""  # noqa: E501
# TODO: FIXME: refine instruction
REFINE_INSTRUCTION: str = """\
<INSTRUCTION>
You will be given the instruction within SUBJECT tag and improvement recommendations within RECOMMENDATIONS tag. Analyze its contents and prepare refined SUBJECT instruction according to described recommendations.
</INSTRUCTION>

<STEPS>
Understand the Original Instructions
- Thoroughly review the SUBJECT instructions, understand their purpose, structure, and key details
- Identify any nuances, requirements, and the intended outcomes
- Think step by step and breakdown its content to individual pieces for better understanding

Analyze Feedback
- Examine improvement recommendations within RECOMMENDATIONS to pinpoint unclear areas, weaknesses, and opportunities for improvement.

Refine and Enhance
- Rewrite the SUBJECT instructions to be concise and highly precise while retaining essential details
- Add necessary context or clarifications to improve usability and understanding
- Preserve all placeholder variables (e.g., `{{variable}}`) exactly as they appear, ensuring their names remain descriptive of their intended content

Format for Readability
- Apply clear, logical formatting and structure to enhance readability and usability.
- Put each key section of the insctucion in capitalized xml tag.
</STEPS>
{guidelines}
<FORMAT>
Provide only the final result containing only the improved instructions without any additional comments.
</FORMAT>
"""  # noqa: E501
