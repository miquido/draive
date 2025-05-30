from collections.abc import Mapping, Sequence
from typing import Any

from haiway import State, ctx, traced

from draive.commons import Meta
from draive.evaluation import (
    EvaluationSuite,
    SuiteEvaluatorResult,
)
from draive.instructions import Instruction, Instructions
from draive.lmm import LMMContext
from draive.multimodal import MultimodalContent
from draive.multimodal.tags import MultimodalTagElement
from draive.parameters import DataModel
from draive.stages import Stage, StageCondition, StageState, stage
from draive.utils import Processing

__all__ = ("refine_instruction",)


class FailureType(State):
    """Classification of evaluation failure types"""

    name: str
    description: str
    refinement_strategy: str


class RefinementCandidate(State):
    """A single refinement candidate with its strategy"""

    content: str
    strategy: str
    reasoning: str


class RefinementAnalysis(State):
    """Analysis of evaluation results with failure classification"""

    failure_types: Sequence[FailureType]
    recommended_strategies: Sequence[str]
    analysis_summary: str


class RefinementHistory(State):
    scores: Sequence[float]
    instructions: Sequence[str]
    improvements: Sequence[float]  # score deltas
    strategies_used: Sequence[str]  # track which strategies were used
    failed_strategies: Sequence[str]  # track strategies that led to worse scores

    @property
    def convergence_trend(self) -> float:
        """Calculate convergence trend over last N rounds"""
        convergence_window = 3
        if len(self.improvements) < convergence_window:
            return float("inf")
        recent = self.improvements[-convergence_window:]
        return sum(recent) / len(recent)

    def is_plateauing(self, threshold: float = 0.001) -> bool:
        """Detect if improvements are diminishing"""
        return abs(self.convergence_trend) < threshold

    @classmethod
    def empty(cls) -> "RefinementHistory":
        """Create empty history for initialization"""
        return cls(
            scores=(), instructions=(), improvements=(), strategies_used=(), failed_strategies=()
        )


@traced
async def refine_instruction[CaseParameters: DataModel, Result: DataModel | str](
    instruction: Instruction,
    /,
    *,
    guidelines: str | None = None,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    rounds_limit: int,
    quality_threshold: float = 0.99,
    convergence_window: int = 3,
    min_improvement_rate: float = 0.001,
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
            convergence_window=convergence_window,
            min_improvement_rate=min_improvement_rate,
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
    history: RefinementHistory
    analysis: RefinementAnalysis | None = None
    candidates: Sequence[RefinementCandidate] = ()


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
                history=RefinementHistory.empty(),
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

        # Enhanced: History tracking and trend analysis
        if processing_state.evaluation is None:
            ctx.log_info("...no previous evaluation using current as a baseline...")
            # First evaluation - establish baseline
            new_history = RefinementHistory(
                scores=(current_evaluation.relative_score,),
                instructions=(current_instruction.content,),
                improvements=(0.0,),
                strategies_used=(),
                failed_strategies=(),
            )
            await Processing.write(
                processing_state.updated(
                    instruction=current_instruction,
                    evaluation=current_evaluation,
                    history=new_history,
                )
            )
            return StageState(
                context=context,
                result=result,
            )

        # Track improvement and detect patterns
        improvement = current_evaluation.relative_score - processing_state.evaluation.relative_score

        # Enhanced: Rollback detection with history awareness
        if improvement <= 0:
            # Check if this is part of a declining pattern
            recent_declines = sum(
                1 for imp in processing_state.history.improvements[-2:] if imp <= 0
            )
            min_history_length = 2
            if (
                recent_declines >= 1
                and len(processing_state.history.improvements) >= min_history_length
            ):
                ctx.log_info(
                    f"Decline pattern detected (improvement: {improvement:.4f}), rolling back..."
                )
                return StageState(
                    context=context,
                    result=MultimodalContent.of(processing_state.instruction.content),
                )
            ctx.log_info(
                f"Minor decline (improvement: {improvement:.4f}), allowing one more attempt..."
            )
        else:
            ctx.log_info(f"Evaluation score improved by {improvement:.4f}, proceeding...")

        # Determine which strategy was used for this refinement
        used_strategy = "UNKNOWN"
        failed_strategies = processing_state.history.failed_strategies

        if hasattr(processing_state, "candidates") and processing_state.candidates:
            # Use the strategy from the first candidate that was selected
            used_strategy = processing_state.candidates[0].strategy

        if improvement <= 0:
            # Add strategy to failed strategies list
            failed_strategies = (*failed_strategies, used_strategy)

        # Update history with new data
        new_history = RefinementHistory(
            scores=(*processing_state.history.scores, current_evaluation.relative_score),
            instructions=(*processing_state.history.instructions, current_instruction.content),
            improvements=(*processing_state.history.improvements, improvement),
            strategies_used=(*processing_state.history.strategies_used, used_strategy),
            failed_strategies=failed_strategies,
        )

        # Continue with updated state
        await Processing.write(
            processing_state.updated(
                instruction=current_instruction,
                evaluation=current_evaluation,
                history=new_history,
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
    convergence_window: int = 3,
    min_improvement_rate: float = 0.001,
) -> Stage:
    return Stage.loop(
        Stage.sequence(
            _refinement_stage(guidelines=guidelines),
            _ensemble_evaluation_stage(evaluation_suite=evaluation_suite),
            # cut out stage context - keep only the result
            Stage.trim_context(limit=0),
        ),
        condition=_refinement_loop_condition(
            rounds_limit=rounds_limit,
            quality_threshold=quality_threshold,
            convergence_window=convergence_window,
            min_improvement_rate=min_improvement_rate,
        ),
    )


def _failure_analysis_stage() -> Stage:
    """Analyze evaluation results and classify failure types"""

    @stage
    async def _failure_analysis(
        *,
        context: LMMContext,
        result: MultimodalContent,
    ) -> StageState:
        ctx.log_info("...analyzing evaluation failures...")
        processing_state: _RefinementState = await Processing.read(
            _RefinementState,
            required=True,
        )
        evaluation: SuiteEvaluatorResult[Any, Any] | None = processing_state.evaluation
        if evaluation is None or not evaluation.cases:
            ctx.log_info("...evaluation results unavailable, skipping analysis...")
            # Return empty analysis for first round
            empty_analysis = RefinementAnalysis(
                failure_types=(),
                recommended_strategies=("CLARITY", "PRECISION"),
                analysis_summary="No evaluation data available - using default strategies",
            )
            await Processing.write(processing_state.updated(analysis=empty_analysis))
            return StageState(context=context, result=result)

        formatted_evaluation_results: MultimodalContent = MultimodalContent.of(
            "\n<EVALUATION_RESULT>",
            evaluation.report(
                include_passed=False,
                include_details=True,
            ),
            "</EVALUATION_RESULT>",
        )

        analysis_response = await Stage.completion(
            formatted_evaluation_results,
            instruction=FAILURE_ANALYSIS_INSTRUCTION,
        ).execute()

        # Parse analysis results
        failure_analysis = MultimodalTagElement.parse_first(
            "FAILURE_ANALYSIS", content=analysis_response
        )
        recommended_strategies = MultimodalTagElement.parse_first(
            "RECOMMENDED_STRATEGIES", content=analysis_response
        )
        analysis_summary = MultimodalTagElement.parse_first(
            "ANALYSIS_SUMMARY", content=analysis_response
        )

        if (
            not all([failure_analysis, recommended_strategies, analysis_summary])
            or failure_analysis is None
            or recommended_strategies is None
            or analysis_summary is None
        ):
            raise ValueError("Incomplete failure analysis response")

        # Extract strategy names from the recommended strategies text
        strategy_text = recommended_strategies.content.to_str()
        strategies = []
        for strategy in [
            "CLARITY",
            "PRECISION",
            "COMPLETENESS",
            "BREVITY",
            "CONTEXTUALIZATION",
            "EXEMPLIFICATION",
        ]:
            if strategy in strategy_text:
                strategies.append(strategy)

        if not strategies:
            strategies = ["CLARITY", "PRECISION"]  # fallback

        analysis = RefinementAnalysis(
            failure_types=(),  # Could be expanded to parse specific failure types
            recommended_strategies=tuple(strategies[:3]),  # Limit to 3 strategies
            analysis_summary=analysis_summary.content.to_str(),
        )

        await Processing.write(processing_state.updated(analysis=analysis))
        return StageState(context=context, result=result)

    return _failure_analysis


def _multi_strategy_refinement_stage(
    *,
    guidelines: str | None,
) -> Stage:
    """Generate multiple refinement candidates using different strategies"""

    @stage
    async def _multi_refinement(
        *,
        context: LMMContext,
        result: MultimodalContent,
    ) -> StageState:
        ctx.log_info("...generating multi-strategy refinement candidates...")
        processing_state: _RefinementState = await Processing.read(
            _RefinementState,
            required=True,
        )

        analysis: RefinementAnalysis | None = getattr(processing_state, "analysis", None)
        if analysis is None:
            # Fallback if no analysis available
            analysis = RefinementAnalysis(
                failure_types=(),
                recommended_strategies=("CLARITY", "PRECISION"),
                analysis_summary="Using default strategies",
            )

        # Prepare context with analysis and instruction
        analysis_context = MultimodalContent.of(
            f"<SUBJECT>{processing_state.instruction.content}</SUBJECT>",
            (
                f"\n<DESCRIPTION>{processing_state.instruction.description}</DESCRIPTION>"
                if processing_state.instruction.description
                else ""
            ),
            "\n<FAILURE_ANALYSIS>",
            analysis.analysis_summary,
            "</FAILURE_ANALYSIS>",
            "\n<RECOMMENDED_STRATEGIES>",
            ", ".join(analysis.recommended_strategies),
            "</RECOMMENDED_STRATEGIES>",
        )

        # Generate multiple candidates
        candidates_response = await Stage.completion(
            analysis_context,
            instruction=MULTI_STRATEGY_REFINEMENT_INSTRUCTION.format(
                guidelines=f"\n<GUIDELINES>{guidelines}</GUIDELINES>\n" if guidelines else "",
            ),
        ).execute()

        # Parse candidates
        candidates = []
        for i in range(1, 4):  # Support up to 3 candidates
            candidate_element = MultimodalTagElement.parse_first(
                f"CANDIDATE_{i}", content=candidates_response
            )
            if candidate_element:
                strategy_elem = MultimodalTagElement.parse_first(
                    "STRATEGY", content=candidate_element.content
                )
                reasoning_elem = MultimodalTagElement.parse_first(
                    "REASONING", content=candidate_element.content
                )
                content_elem = MultimodalTagElement.parse_first(
                    "CONTENT", content=candidate_element.content
                )

                if (
                    all([strategy_elem, reasoning_elem, content_elem])
                    and strategy_elem is not None
                    and reasoning_elem is not None
                    and content_elem is not None
                ):
                    candidates.append(
                        RefinementCandidate(
                            content=content_elem.content.to_str().strip(),
                            strategy=strategy_elem.content.to_str().strip(),
                            reasoning=reasoning_elem.content.to_str().strip(),
                        )
                    )

        if not candidates:
            raise ValueError("No valid refinement candidates generated")

        ctx.log_info(f"...generated {len(candidates)} refinement candidates...")

        # Store candidates for evaluation and return first candidate for now
        await Processing.write(processing_state.updated(candidates=candidates))
        return StageState(
            context=context,
            result=MultimodalContent.of(candidates[0].content),
        )

    return _multi_refinement


def _ensemble_evaluation_stage[CaseParameters: DataModel, Result: DataModel | str](
    *,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
) -> Stage:
    """Evaluate multiple refinement candidates and select the best one"""

    @stage
    async def _ensemble_eval(
        *,
        context: LMMContext,
        result: MultimodalContent,
    ) -> StageState:
        ctx.log_info("...evaluating refinement candidates...")
        processing_state: _RefinementState = await Processing.read(
            _RefinementState,
            required=True,
        )

        if not hasattr(processing_state, "candidates") or not processing_state.candidates:
            # No candidates to evaluate, return as-is
            return StageState(context=context, result=result)

        candidates = processing_state.candidates
        ctx.log_info(f"...evaluating {len(candidates)} candidates...")

        best_candidate = None
        best_score = -1.0
        best_evaluation = None

        # Evaluate each candidate
        for i, candidate in enumerate(candidates):
            ctx.log_info(f"...evaluating candidate {i + 1} ({candidate.strategy})...")

            # Create temporary instruction with candidate content
            candidate_instruction = processing_state.instruction.updated(content=candidate.content)

            # Evaluate this candidate
            with ctx.updated(_instructions(instruction=candidate_instruction)):
                candidate_evaluation = await evaluation_suite()

            ctx.log_info(
                f"...candidate {i + 1} score: {candidate_evaluation.relative_score:.4f} "
                f"(strategy: {candidate.strategy})..."
            )

            # Track best candidate
            if candidate_evaluation.relative_score > best_score:
                best_candidate = candidate
                best_score = candidate_evaluation.relative_score
                best_evaluation = candidate_evaluation

        if best_candidate is None:
            raise ValueError("No valid candidate found during ensemble evaluation")

        ctx.log_info(
            f"...selected best candidate with score {best_score:.4f} "
            f"(strategy: {best_candidate.strategy})..."
        )

        # Update state with best candidate and its evaluation
        best_instruction = processing_state.instruction.updated(content=best_candidate.content)
        await Processing.write(
            processing_state.updated(
                instruction=best_instruction,
                evaluation=best_evaluation,
            )
        )

        return StageState(
            context=context,
            result=MultimodalContent.of(best_candidate.content),
        )

    return _ensemble_eval


def _refinement_stage(
    *,
    guidelines: str | None,
) -> Stage:
    """Enhanced refinement stage with failure analysis and multi-strategy generation"""
    return Stage.sequence(
        _failure_analysis_stage(),
        _multi_strategy_refinement_stage(guidelines=guidelines),
    )


def _refinement_loop_condition[CaseParameters: DataModel, Result: DataModel | str](
    *,
    rounds_limit: int,
    quality_threshold: float,
    convergence_window: int = 3,
    min_improvement_rate: float = 0.001,
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
                history=RefinementHistory.empty(),
            ),
        )
        ctx.log_info(f"...ending refinement round {processing_state.round}...")

        # Standard termination conditions
        if processing_state.round >= rounds_limit:
            ctx.log_info("...refinement round limit reached...")
            return False

        if (
            processing_state.evaluation is not None
            and processing_state.evaluation.relative_score >= quality_threshold
        ):
            ctx.log_info("...refinement quality threshold reached...")
            return False

        # Enhanced: Convergence detection
        if (
            processing_state.round >= convergence_window
            and len(processing_state.history.improvements) >= convergence_window
        ):
            recent_improvements = processing_state.history.improvements[-convergence_window:]
            avg_improvement = sum(recent_improvements) / len(recent_improvements)

            if abs(avg_improvement) < min_improvement_rate:
                ctx.log_info(
                    f"...convergence detected: avg improvement {avg_improvement:.6f} "
                    f"< {min_improvement_rate}..."
                )
                return False

            ctx.log_info(
                f"...continuing refinement: avg improvement {avg_improvement:.6f} "
                f">= {min_improvement_rate}..."
            )

        await Processing.write(processing_state.updated(round=processing_state.round + 1))
        return True

    return condition


FAILURE_ANALYSIS_INSTRUCTION: str = """\
You are an expert evaluation analyst specializing in instruction failure classification and \
targeted improvement strategies.

<TASK>
Analyze the provided evaluation results and classify the types of failures observed. \
Based on your analysis, recommend specific refinement strategies that would address the \
identified issues.
</TASK>

<FAILURE_TYPES>
- COMPREHENSION: The instruction is unclear, ambiguous, or difficult to understand
- PRECISION: The instruction lacks specific details, constraints, or clear boundaries
- COMPLETENESS: The instruction has gaps in coverage or missing information
- EFFICIENCY: The instruction is verbose, redundant, or poorly structured
- CONTEXT: The instruction doesn't provide adequate context or background
- EXAMPLES: The instruction lacks helpful examples or demonstrations
</FAILURE_TYPES>

<REFINEMENT_STRATEGIES>
- CLARITY: Improve language clarity, remove ambiguity, restructure for better flow
- PRECISION: Add specific details, constraints, measurements, and clear boundaries
- COMPLETENESS: Fill information gaps, add missing steps or considerations
- BREVITY: Remove redundancy, consolidate information, improve conciseness
- CONTEXTUALIZATION: Add background information, situational context, purpose
- EXEMPLIFICATION: Include examples, demonstrations, or sample outputs
</REFINEMENT_STRATEGIES>

<OUTPUT_FORMAT>
Provide your analysis in the following format:

<FAILURE_ANALYSIS>
[List the primary failure types observed and their evidence from the evaluation]
</FAILURE_ANALYSIS>

<RECOMMENDED_STRATEGIES>
[List 2-3 specific refinement strategies that would best address the identified failures]
</RECOMMENDED_STRATEGIES>

<ANALYSIS_SUMMARY>
[Brief summary of key issues and improvement approach]
</ANALYSIS_SUMMARY>
</OUTPUT_FORMAT>
"""

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

MULTI_STRATEGY_REFINEMENT_INSTRUCTION: str = """\
You are an expert instruction optimizer specializing in generating multiple targeted \
refinement strategies. Your role is to create 2-3 different refinement candidates, each \
using a distinct approach to address the identified issues.

<INSTRUCTIONS>
You will receive:
1. An instruction labeled as SUBJECT that needs refinement
2. Analysis results showing specific failure types and recommended strategies
3. Optional guidelines for refinement

Your task is to generate 2-3 refinement candidates, each following a different strategy \
from the recommended approaches.
</INSTRUCTIONS>

<STRATEGY_DEFINITIONS>
- CLARITY: Focus on improving language clarity, removing ambiguity, and restructuring \
for better flow
- PRECISION: Emphasize adding specific details, constraints, measurements, and clear boundaries
- COMPLETENESS: Concentrate on filling information gaps and adding missing steps or considerations
- BREVITY: Prioritize removing redundancy, consolidating information, and improving conciseness
- CONTEXTUALIZATION: Add background information, situational context, and purpose explanation
- EXEMPLIFICATION: Include examples, demonstrations, or sample outputs to aid understanding
</STRATEGY_DEFINITIONS>

<OUTPUT_FORMAT>
For each refinement candidate, provide:

<CANDIDATE_1>
<STRATEGY>[Strategy name from recommended strategies]</STRATEGY>
<REASONING>[Brief explanation of how this strategy addresses the identified issues]</REASONING>
<CONTENT>
[The refined instruction following this specific strategy]
</CONTENT>
</CANDIDATE_1>

<CANDIDATE_2>
<STRATEGY>[Different strategy name]</STRATEGY>
<REASONING>[Brief explanation of approach]</REASONING>
<CONTENT>
[The refined instruction following this strategy]
</CONTENT>
</CANDIDATE_2>

[Include CANDIDATE_3 if three strategies were recommended]
</OUTPUT_FORMAT>

<REQUIREMENTS>
- Each candidate must use a different strategy approach
- Maintain the original intent and core purpose of the instruction
- Ensure each candidate stands alone as a complete, usable instruction
- Focus on the specific issues identified in the failure analysis
{guidelines}
</REQUIREMENTS>
"""
