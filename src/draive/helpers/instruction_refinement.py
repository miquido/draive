from collections.abc import Mapping, Sequence
from typing import Any

from haiway import State, ctx, traced

from draive.evaluation import EvaluationSuite, SuiteEvaluatorResult
from draive.instructions import Instruction, Instructions
from draive.multimodal import MultimodalContent, MultimodalTagElement
from draive.parameters import DataModel
from draive.stages import Stage, StageLoopConditioning, StageState, stage

__all__ = ("refine_instruction",)


class _RefinementCandidate(State):
    strategy: str
    reasoning: str
    content: str


class _RefinementAnalysis(State):
    recommended_strategies: Sequence[str]
    analysis_summary: str


class _RefinementHistoryElement(State):
    instruction: Instruction
    evaluation: SuiteEvaluatorResult[Any, Any]
    strategy: str
    score: float
    improvement: float


class _RefinementHistory(State):
    elements: Sequence[_RefinementHistoryElement] = ()
    failed_strategies: Sequence[str] = ()

    def convergence_trend(
        self,
        *,
        window: int,
    ) -> float:
        if len(self.elements) < window:
            return float("inf")

        recent: Sequence[float] = [element.improvement for element in self.elements[-window:]]
        return sum(recent) / len(recent)

    def is_plateauing(
        self,
        *,
        window: int,
        threshold: float,
    ) -> bool:
        return abs(self.convergence_trend(window=window)) < threshold


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
    assert convergence_window > 0  # nosec: B101
    assert 1 >= quality_threshold >= 0  # nosec: B101
    assert 1 >= min_improvement_rate >= 0  # nosec: B101
    ctx.log_info("Preparing instruction refinement...")

    result: MultimodalContent = await Stage.sequence(
        _initialization_stage(
            instruction=instruction,
            guidelines=guidelines,
            evaluation_suite=evaluation_suite,
        ),
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
    evaluation: SuiteEvaluatorResult[Any, Any]
    history: _RefinementHistory
    analysis: _RefinementAnalysis
    candidates: Sequence[_RefinementCandidate]


def _initialization_stage[CaseParameters: DataModel, Result: DataModel | str](
    *,
    instruction: Instruction,
    guidelines: str | None,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
) -> Stage:
    @stage
    async def initialization(
        *,
        state: StageState,
    ) -> StageState:
        ctx.log_info("...evaluating initial instruction...")
        evaluation: SuiteEvaluatorResult[CaseParameters, Result]
        with ctx.updated(_instructions(instruction=instruction)):
            evaluation = await evaluation_suite()

        ctx.log_info("...using initial evaluation as a baseline...")
        return state.updated(
            _RefinementState(
                instruction=instruction,
                evaluation=evaluation,
                history=_RefinementHistory(
                    elements=(
                        _RefinementHistoryElement(
                            instruction=instruction,
                            evaluation=evaluation,
                            strategy="initial",
                            score=evaluation.relative_score,
                            improvement=0.0,
                        ),
                    ),
                    failed_strategies=(),
                ),
                analysis=_RefinementAnalysis(
                    recommended_strategies=(),
                    analysis_summary="N/A",
                ),
                candidates=(),
            ),
            # set the result to be initial instruction
            result=MultimodalContent.of(instruction.content),
        )

    return initialization


def _refinement_loop_stage[CaseParameters: DataModel, Result: DataModel | str](
    *,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
    guidelines: str | None,
    rounds_limit: int,
    quality_threshold: float,
    convergence_window: int,
    min_improvement_rate: float,
) -> Stage:
    return Stage.loop(
        Stage.sequence(
            _refinement_stage(guidelines=guidelines),
            _ensemble_evaluation_stage(evaluation_suite=evaluation_suite),
            # cut out stage context - keep only the result
            Stage.trim_context(limit=0),
        ),
        mode="pre_check",
        condition=_refinement_loop_condition(
            rounds_limit=rounds_limit,
            quality_threshold=quality_threshold,
            convergence_window=convergence_window,
            min_improvement_rate=min_improvement_rate,
        ),
    )


def _failure_analysis_stage() -> Stage:
    @stage
    async def _failure_analysis(
        *,
        state: StageState,
    ) -> StageState:
        ctx.log_info("...analyzing evaluation failures...")
        refinement_state: _RefinementState = state.get(
            _RefinementState,
            required=True,
        )

        evaluation: SuiteEvaluatorResult[Any, Any] = refinement_state.evaluation
        if not evaluation.cases:
            raise ValueError("Evaluation results unavailable, can't refine instruction")

        analysis_response: MultimodalContent = await Stage.completion(
            MultimodalContent.of(
                "\n<EVALUATION_RESULT>",
                evaluation.report(
                    include_passed=False,
                    include_details=True,
                ),
                "</EVALUATION_RESULT>",
            ),
            instruction=FAILURE_ANALYSIS_INSTRUCTION,
        ).execute()

        recommended_strategies: set[str] = set()
        if strategies := MultimodalTagElement.parse_first(
            "RECOMMENDED_STRATEGIES",
            content=analysis_response,
        ):
            for strategy in strategies.content.to_str().split(","):
                recommended_strategies.add(strategy.strip().upper())

            if not recommended_strategies:
                raise ValueError("Evaluation analysis unavailable, can't refine instruction")

        else:
            raise ValueError(
                "Incomplete failure analysis response"
                f" - missing recommendation:\n{analysis_response}"
            )

        analysis_summary: str
        if summary := MultimodalTagElement.parse_first(
            "ANALYSIS_SUMMARY",
            content=analysis_response,
        ):
            analysis_summary = summary.content.to_str()

        else:
            raise ValueError(
                f"Incomplete failure analysis response - missing summary:\n{analysis_response}"
            )

        return state.updated(
            refinement_state.updated(
                analysis=_RefinementAnalysis(
                    recommended_strategies=tuple(recommended_strategies),
                    analysis_summary=analysis_summary,
                ),
            ),
        )

    return _failure_analysis


def _multi_strategy_refinement_stage(
    *,
    guidelines: str | None,
) -> Stage:
    @stage
    async def _multi_refinement(
        *,
        state: StageState,
    ) -> StageState:
        ctx.log_info("...generating multi-strategy refinement candidates...")
        refinement_state: _RefinementState = state.get(
            _RefinementState,
            required=True,
        )

        analysis: _RefinementAnalysis = refinement_state.analysis

        # Prepare context with analysis and instruction
        analysis_context = MultimodalContent.of(
            f"<SUBJECT>{refinement_state.instruction.content}</SUBJECT>",
            (
                f"\n<DESCRIPTION>{refinement_state.instruction.description}</DESCRIPTION>"
                if refinement_state.instruction.description
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
        candidates: list[_RefinementCandidate] = []
        for candidate_element in MultimodalTagElement.parse(
            "CANDIDATE",
            content=candidates_response,
        ):
            strategy_elem: MultimodalTagElement | None = MultimodalTagElement.parse_first(
                "STRATEGY",
                content=candidate_element.content,
            )
            if strategy_elem is None:
                continue

            reasoning_elem: MultimodalTagElement | None = MultimodalTagElement.parse_first(
                "REASONING",
                content=candidate_element.content,
            )
            if reasoning_elem is None:
                continue

            content_elem: MultimodalTagElement | None = MultimodalTagElement.parse_first(
                "CONTENT",
                content=candidate_element.content,
            )
            if content_elem is None:
                continue

            candidates.append(
                _RefinementCandidate(
                    content=content_elem.content.to_str().strip(),
                    strategy=strategy_elem.content.to_str().strip(),
                    reasoning=reasoning_elem.content.to_str().strip(),
                )
            )

        if not candidates:
            raise ValueError("No valid refinement candidates generated")

        ctx.log_info(f"...generated {len(candidates)} refinement candidates...")

        # Store candidates for evaluation and return first candidate for now
        return state.updated(
            refinement_state.updated(
                candidates=candidates,
            ),
            result=MultimodalContent.of(candidates[0].content),
        )

    return _multi_refinement


def _ensemble_evaluation_stage[CaseParameters: DataModel, Result: DataModel | str](
    *,
    evaluation_suite: EvaluationSuite[CaseParameters, Result],
) -> Stage:
    @stage
    async def _ensemble_eval(
        *,
        state: StageState,
    ) -> StageState:
        ctx.log_info("...evaluating refinement candidates...")
        refinement_state: _RefinementState = state.get(
            _RefinementState,
            required=True,
        )

        if not refinement_state.candidates:
            # No candidates to evaluate, return as-is
            return state

        candidates: Sequence[_RefinementCandidate] = refinement_state.candidates
        ctx.log_info(f"...evaluating {len(candidates)} candidates...")

        best_candidate: _RefinementCandidate | None = None
        best_score: float = -1.0
        best_evaluation: SuiteEvaluatorResult[Any, Any] | None = None

        # Evaluate each candidate
        for i, candidate in enumerate(candidates):
            ctx.log_info(f"...evaluating candidate {i + 1} ({candidate.strategy})...")

            # Create temporary instruction with candidate content
            candidate_instruction: Instruction = refinement_state.instruction.updated(
                content=candidate.content
            )

            # Evaluate this candidate
            with ctx.updated(_instructions(instruction=candidate_instruction)):
                candidate_evaluation: SuiteEvaluatorResult[Any, Any] = await evaluation_suite()

            ctx.log_info(
                f"...candidate {i + 1} score: {candidate_evaluation.relative_score:.4f} "
                f"(strategy: {candidate.strategy})..."
            )

            # Track best candidate
            if candidate_evaluation.relative_score > best_score:
                best_candidate = candidate
                best_score = candidate_evaluation.relative_score
                best_evaluation = candidate_evaluation

        if best_candidate is None or best_evaluation is None:
            raise ValueError("No valid candidate found during ensemble evaluation")

        ctx.log_info(
            f"...selected best candidate with score {best_score:.4f} "
            f"(strategy: {best_candidate.strategy})..."
        )

        best_instruction: Instruction = refinement_state.instruction.updated(
            content=best_candidate.content
        )

        return state.updated(
            # Update state with best candidate and its evaluation
            refinement_state.updated(
                instruction=best_instruction,
                evaluation=best_evaluation,
                history=_RefinementHistory(
                    elements=(
                        *refinement_state.history.elements,
                        _RefinementHistoryElement(
                            instruction=best_instruction,
                            evaluation=best_evaluation,
                            strategy=best_candidate.strategy,
                            score=best_evaluation.relative_score,
                            improvement=best_evaluation.relative_score
                            - refinement_state.evaluation.relative_score,
                        ),
                    ),
                    failed_strategies=refinement_state.history.failed_strategies,
                ),
            ),
            result=MultimodalContent.of(best_instruction.content),
        )

    return _ensemble_eval


def _refinement_stage(
    *,
    guidelines: str | None,
) -> Stage:
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
) -> StageLoopConditioning:
    async def condition(
        *,
        state: StageState,
        iteration: int,
    ) -> bool:
        refinement_state: _RefinementState = state.get(
            _RefinementState,
            required=True,
        )
        ctx.log_info(f"...refinement round {iteration}...")

        # Standard termination conditions
        if iteration >= rounds_limit:
            ctx.log_info("...refinement round limit reached...")
            return False

        if not refinement_state.evaluation.cases:
            ctx.log_info("...evaluation results empty...")
            return False

        if refinement_state.evaluation.relative_score >= quality_threshold:
            ctx.log_info("...refinement quality threshold reached...")
            return False

        # Convergence detection
        if len(refinement_state.history.elements) >= convergence_window:
            history: Sequence[_RefinementHistoryElement] = refinement_state.history.elements[
                -convergence_window:
            ]
            avg_improvement: float = sum(element.improvement for element in history) / len(history)

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
- SUBJECT: The instruction doesn't describe goal or expected results correctly
- EXAMPLES: The instruction lacks helpful examples or demonstrations
</FAILURE_TYPES>

<REFINEMENT_STRATEGIES>
- CLARITY: Improve language clarity, remove ambiguity, restructure for better flow
- PRECISION: Add specific details, constraints, measurements, and clear boundaries
- COMPLETENESS: Fill information gaps, add missing steps or considerations
- BREVITY: Remove redundancy, consolidate information, improve conciseness
- CONTEXTUALIZATION: Add background information, situational context, purpose
- EXPLANATION: Add task and expected results description, explain the subject
- EXEMPLIFICATION: Include examples, demonstrations, or sample outputs
</REFINEMENT_STRATEGIES>

<OUTPUT_FORMAT>
Provide your analysis by wrapping its parts in xml tags in the following format:

<FAILURE_ANALYSIS>
[List the primary failure types observed and their evidence from the evaluation]
</FAILURE_ANALYSIS>

<RECOMMENDED_STRATEGIES>
[List 2-3, comma separated specific refinement strategies that would best address\
 the identified failures]
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
- EXPLANATION: Add task and expected results description, explain the subject
- EXEMPLIFICATION: Include examples, demonstrations, or sample outputs to aid understanding
</STRATEGY_DEFINITIONS>

<OUTPUT_FORMAT>
For each refinement candidate, provide:

<CANDIDATE>
<STRATEGY>[Strategy name from recommended strategies]</STRATEGY>
<REASONING>[Brief explanation of how this strategy addresses the identified issues]</REASONING>
<CONTENT>
[The refined instruction following this specific strategy]
</CONTENT>
</CANDIDATE>

<CANDIDATE>
<STRATEGY>[Different strategy name]</STRATEGY>
<REASONING>[Brief explanation of approach]</REASONING>
<CONTENT>
[The refined instruction following this strategy]
</CONTENT>
</CANDIDATE>

[Include more CANDIDATE tags if more strategies were recommended]
</OUTPUT_FORMAT>

<REQUIREMENTS>
- Each candidate must use a different strategy approach
- Maintain the original intent and core purpose of the instruction
- Ensure each candidate stands alone as a complete, usable instruction
- Focus on the specific issues identified in the failure analysis
- Optimize for LLM system prompt usage
{guidelines}
</REQUIREMENTS>
"""
