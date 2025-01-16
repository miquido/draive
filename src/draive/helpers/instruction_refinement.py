from collections.abc import Mapping
from typing import Any

from haiway import State, ctx, traced

from draive.evaluation import (
    EvaluationSuite,
    SuiteEvaluatorResult,
)
from draive.instructions import Instruction, Instructions
from draive.lmm import LMMContext
from draive.lmm.types import LMMCompletion, LMMInput
from draive.multimodal import MultimodalContent
from draive.multimodal.tags import MultimodalTagElement
from draive.parameters import DataModel
from draive.stages import Stage, StageCondition
from draive.utils import Processing

__all__ = [
    "refine_instruction",
]


@traced
async def refine_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: Instruction,
    /,
    *,
    guidelines: str | None = None,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    rounds_limit: int,
) -> Instruction:
    assert rounds_limit > 0  # nosec: B101
    ctx.log_info("Preparing instruction refinement...")

    result: MultimodalContent = await Stage.sequence(
        _initialization_stage(
            instruction=instruction,
            guidelines=guidelines,
        ),
        _evaluation_stage(evaluation_suite=evaluation_suite),
        _refinement_loop_stage(
            evaluation_suite=evaluation_suite,
            guidelines=guidelines,
            rounds_limit=rounds_limit,
        ),
    ).execute()
    ctx.log_info("...instruction refinement finished!")

    return instruction.updated(content=result.as_string())


def _instructions(
    *,
    instruction: Instruction,
) -> Instructions:
    instructions: Instructions = ctx.state(Instructions)

    async def instruction_fetch(
        name: str,
        /,
        *,
        arguments: Mapping[str, str | float | int] | None = None,
        **extra: Any,
    ) -> Instruction | None:
        if name == instruction.name:
            if arguments:
                return instruction.updated(arguments={**instruction.arguments, **arguments})

            else:
                return instruction

        else:  # preserve other prompts unchanged
            return await instructions.fetch(
                name,
                arguments=arguments,
                **extra,
            )

    return Instructions(fetching=instruction_fetch)


class _RefinementState(State):
    instruction: Instruction
    evaluation: SuiteEvaluatorResult[Any, Any] | None
    round: int


def _initialization_stage(
    *,
    instruction: Instruction,
    guidelines: str | None,
) -> Stage:
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

    return Stage.transform_result(initialization)


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
        with ctx.updated(_instructions(instruction=current_instruction)):
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
            Stage.trim_context(limit=0),
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


REVIEW_INSTRUCTION: str = """\
<INSTRUCTION>
Analyze the content within the EVALUATION_RESULT tag, focusing on the reported results of the evaluation suite. Identify key performance metrics, strengths, weaknesses, and any areas where the process or system under evaluation can be enhanced. Based on this analysis, formulate detailed conclusions that highlight specific issues and potential improvements. Ensure that recommendations directly address the instruction or process being evaluated, aiming to optimize its effectiveness, efficiency, and reliability. Present your findings and suggestions in a clear, structured manner suitable for guiding subsequent development or refinement efforts.
</INSTRUCTION>
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
Refine the given SUBJECT instruction to ensure clarity, conciseness, and precision. Use a clear, actionable language that accurately reflects the task requirements. Structure the instruction logically, emphasizing the key steps or actions needed. Replace vague terms with specific descriptors where applicable and verify that the instruction is easily understandable and directly aligned with the intended outcome.
</INSTRUCTION>
{guidelines}
<FORMAT>
Provide only the final result containing only the improved instructions without any additional comments.
</FORMAT>
"""  # noqa: E501
