from collections.abc import Mapping
from typing import Any

from haiway import State, ctx, traced

from draive.commons import Meta
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
from draive.stages import Stage, StageCondition, StageState, stage
from draive.utils import Processing

__all__ = ("refine_instruction",)


@traced
async def refine_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: Instruction,
    /,
    *,
    guidelines: str | None = None,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    rounds_limit: int,
    quality_threshold: float = 0.99,
) -> Instruction:
    assert rounds_limit > 0  # nosec: B101
    assert 1 >= quality_threshold >= 0  # nosec: B101
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
            quality_threshold=quality_threshold,
        ),
    ).execute()
    ctx.log_info("...instruction refinement finished!")

    return instruction.updated(content=result.to_str())


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
    @stage
    async def evaluation(
        context: LMMContext,
        result: MultimodalContent,
    ) -> StageState:
        processing_state: _RefinementState = await Processing.read(
            _RefinementState,
            required=True,
        )
        current_instruction: Instruction = processing_state.instruction.updated(
            content=result.to_str()
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
            return StageState(
                context=context,
                result=result,
            )

        if current_evaluation.relative_score <= previous_evaluation.relative_score:
            ctx.log_info("...evaluation score worse, rolling back to previous instruction...")
            # rollback to previous instruction - processing state holds previous one
            return StageState(
                context=context,
                result=MultimodalContent.of(processing_state.instruction.content),
            )

        ctx.log_info("...evaluation score better, proceeding...")
        # continue with improved state otherwise
        await Processing.write(
            processing_state.updated(
                instruction=current_instruction,
                evaluation=current_evaluation,
            )
        )
        return StageState(
            context=context,
            result=result,
        )

    return evaluation


def _refinement_loop_stage[CaseParameters: DataModel, Result: DataModel | str](
    *,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    guidelines: str | None,
    rounds_limit: int,
    quality_threshold: float,
) -> Stage:
    return Stage.loop(
        Stage.sequence(
            _refinement_stage(guidelines=guidelines),
            _evaluation_stage(evaluation_suite=evaluation_suite),
            # cut out stage context - keep only the result
            Stage.trim_context(limit=0),
        ),
        condition=_refinement_loop_condition(
            rounds_limit=rounds_limit,
            quality_threshold=quality_threshold,
        ),
    )


def _refinement_stage(
    *,
    guidelines: str | None,
) -> Stage:
    @stage
    async def _analysis(
        *,
        context: LMMContext,
        result: MultimodalContent,
    ) -> StageState:
        ctx.log_info("...analyzing evaluation results...")
        processing_state: _RefinementState = await Processing.read(
            _RefinementState,
            required=True,
        )
        evaluation: SuiteEvaluatorResult[Any, Any] | None = processing_state.evaluation
        if evaluation is None or not evaluation.cases:
            ctx.log_info("...evaluation results unavailable, skipping analysis...")
            # nothing to analyze if no evaluation was done
            return StageState(
                context=context,
                result=result,
            )

        formatted_evaluation_results: MultimodalContent = MultimodalContent.of(
            "\n<EVALUATION_RESULT>",
            evaluation.report(
                include_passed=False,
                include_details=True,
            ),
            "</EVALUATION_RESULT>",
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

        return StageState(
            context=(
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
            result=result,
        )

    return Stage.sequence(
        _analysis,
        Stage.loopback_completion(
            instruction=REFINE_INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>{guidelines}</GUIDELINES>\n" if guidelines else "",
            ),
        ),
    )


def _refinement_loop_condition[CaseParameters: DataModel, Result: DataModel | str](
    *,
    rounds_limit: int,
    quality_threshold: float,
) -> StageCondition:
    async def condition(
        *,
        meta: Meta,
        context: LMMContext,
        result: MultimodalContent,
    ) -> bool:
        processing_state: _RefinementState = await Processing.read(
            _RefinementState,
            default=_RefinementState(
                instruction=Instruction.of(result.to_str()),
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
            and processing_state.evaluation.relative_score >= quality_threshold
        ):
            ctx.log_info("...refinement evaluation passed...")
            return False

        await Processing.write(processing_state.updated(round=processing_state.round + 1))
        return True

    return condition


REVIEW_INSTRUCTION: str = """\
You are an evaluation analyst specializing in performance assessment and process optimization. Your role is to thoroughly analyze evaluation results and provide actionable improvement recommendations.

<INSTRUCTIONS>
You will be given evaluation results within an EVALUATION_RESULT tag that need to be analyzed. Focus on identifying performance metrics, strengths, weaknesses, and opportunities for enhancement. When the EVALUATION_RESULT tag is empty or missing, respond with "<RECOMMENDATIONS>No evaluation results provided for analysis.</RECOMMENDATIONS>".
</INSTRUCTIONS>

<DETAILED_TASK_DESCRIPTION>
Analyze the provided evaluation results by:
1. Identifying and extracting key performance metrics
2. Cataloging strengths and areas where the evaluated process/system performs well
3. Pinpointing weaknesses, failures, or underperforming areas
4. Detecting patterns, trends, or recurring issues across the evaluation data
5. Assessing the overall effectiveness, efficiency, and reliability of the evaluated process/system

Based on your analysis, formulate specific, actionable recommendations that:
- Directly address identified weaknesses and issues
- Build upon existing strengths
- Optimize the effectiveness of the instruction or process being evaluated
- Improve efficiency and reliability
- Are practical and implementable
- Avoids binding to concrete cases but forms general performance improvement suggestions
</DETAILED_TASK_DESCRIPTION>

<ANALYSIS_PROCESS>
Before providing your recommendations:
1. Thoroughly review all evaluation data and results
2. Categorize findings into strengths, weaknesses, and neutral observations
3. Prioritize issues based on their impact on overall performance
4. Consider the context and purpose of the evaluated process/system
5. Develop specific, measurable improvements for each identified issue
6. Ensure recommendations are directly relevant to the evaluation findings
</ANALYSIS_PROCESS>

<OUTPUT_FORMAT>
Structure your analysis as follows:
1. Brief summary of key findings
2. Detailed breakdown of:
   - Performance metrics observed
   - Identified strengths
   - Identified weaknesses
   - Critical issues requiring immediate attention
3. Comprehensive recommendations addressing each weakness and opportunity for improvement
4. Prioritization of recommendations (if applicable)

Place your final improvement recommendations within a RECOMMENDATIONS tag:
<RECOMMENDATIONS>
[Your detailed, actionable improvement recommendations here]
</RECOMMENDATIONS>
</OUTPUT_FORMAT>
"""  # noqa: E501
REFINE_INSTRUCTION: str = """\
You are an expert instruction optimizer specializing in refining and improving existing instructions for maximum clarity and effectiveness.
Your role is to analyze and enhance instructions by making them clearer, more concise, and more actionable. Focus on eliminating ambiguity while preserving the original intent and ensuring the refined version directly supports the intended outcome.

<INSTRUCTIONS>
You will receive an instruction labeled as SUBJECT that needs refinement. Your task is to improve this instruction following these specific requirements:

1. **Clarity Enhancement**
    - Replace vague or ambiguous terms with specific, concrete language
    - Ensure each sentence conveys exactly one clear idea
    - Remove any potential for misinterpretation
    - always use English language for the instruction description

2. **Conciseness Optimization**
    - Eliminate redundant words and phrases
    - Combine related ideas efficiently
    - Maintain brevity without sacrificing essential information

3. **Precision Improvement**
    - Use actionable verbs and specific descriptors
    - Define any technical terms or concepts if necessary
    - Ensure measurements, quantities, or specifications are exact
    - Use placeholders for available variables where their content is appropriate refering to them by their names (e.g., `{{variable}}`)
    - Do not add any variables when there are none available
    - Variables will be resolved to actual content like {{time}} will be replaced with the actual time

4. **Logical Structure**
    - Organize steps in sequential order
    - Group related actions together
    - Use consistent formatting and numbering
    - Use xml tags (including nested) with snake case names to wrap sections and separate important parts of the instruction
</INSTRUCTIONS>

<REFINEMENT_PROCESS>
Before providing the refined instruction:
1. Identify the core objective of the original instruction
2. List all key actions or steps required
3. Detect vague terms that need specific replacements
4. Determine the most logical flow of information
5. Verify alignment with the intended outcome
</REFINEMENT_PROCESS>
{guidelines}
<OUTPUT_REQUIREMENTS>
Provide only the refined instruction text without any explanatory comments, analysis, or metadata. The refined instruction should stand alone as a complete, improved version of the original.
</OUTPUT_REQUIREMENTS>
"""  # noqa: E501
