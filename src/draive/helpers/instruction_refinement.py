import random
from asyncio import gather
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID, uuid4

from haiway import State, as_dict, ctx, traced

from draive.evaluation import EvaluationSuite, SuiteEvaluatorResult
from draive.evaluation.suite import EvaluationSuiteCase
from draive.instructions import Instruction, Instructions
from draive.multimodal import MultimodalContent, MultimodalTagElement
from draive.parameters import DataModel
from draive.stages import Stage, StageState, stage

__all__ = ("refine_instruction",)


@traced
async def refine_instruction[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
    Value,
](
    instruction: Instruction,
    /,
    *,
    guidelines: str | None = None,
    evaluation_suite: EvaluationSuite[SuiteParameters, CaseParameters, Value],
    rounds_limit: int,
    sample_ratio: float = 0.1,
    candidates_limit: int = 3,
    performance_drop_threshold: float = 0.5,
    quality_threshold: float = 0.99,
) -> Instruction:
    """
    Refine instruction using binary tree exploration with performance pruning.

    Each node generates exactly 2 complementary strategies.
    Branches are pruned if performance drops below threshold.

    Args:
        instruction: The instruction to refine
        guidelines: Optional guidelines for refinement
        evaluation_suite: Suite to evaluate instruction performance
        rounds_limit: Maximum depth of refinement tree
        sample_ratio: Fraction of passing cases to include in focused evaluation
        candidates_limit: Number of top candidates to fully evaluate
        performance_drop_threshold: Prune branches with score drop > this threshold
        quality_threshold: Stop if score reaches this threshold
    """

    assert rounds_limit > 0  # nosec: B101
    assert 1 >= sample_ratio > 0  # nosec: B101
    assert candidates_limit > 0  # nosec: B101
    assert 1 >= performance_drop_threshold > 0  # nosec: B101
    assert 1 >= quality_threshold >= 0  # nosec: B101

    ctx.log_info(
        f"Starting tree-based refinement: "
        f"max_depth={rounds_limit}, "
        f"sample_ratio={sample_ratio}, "
        f"performance_drop_threshold={performance_drop_threshold}"
    )

    result: MultimodalContent = await Stage.sequence(
        _tree_initialization_stage(
            instruction=instruction,
            evaluation_suite=evaluation_suite,
            sample_ratio=sample_ratio,
            performance_drop_threshold=performance_drop_threshold,
            guidelines=guidelines,
            rounds_limit=rounds_limit,
        ),
        _tree_exploration_stage(
            evaluation_suite=evaluation_suite,
            rounds_limit=rounds_limit,
            quality_threshold=quality_threshold,
        ),
        _tree_finalization_stage(
            evaluation_suite=evaluation_suite,
            quality_threshold=quality_threshold,
            candidates_limit=candidates_limit,
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


class _RefinementTreeNode[SuiteParameters: DataModel, CaseParameters: DataModel](State):
    identifier: UUID
    instruction: Instruction
    strategy: str
    parent_id: UUID | None
    depth: int
    focused_evaluation: SuiteEvaluatorResult[SuiteParameters, CaseParameters]
    complete_evaluation: SuiteEvaluatorResult[SuiteParameters, CaseParameters] | None
    children: Sequence[UUID]
    pruned: bool

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def focused_evaluation_score(self) -> float:
        return self.focused_evaluation.relative_score

    @property
    def complete_evaluation_score(self) -> float | None:
        if self.complete_evaluation is None:
            return None

        return self.complete_evaluation.relative_score


class _RefinementState[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
](State):
    root: _RefinementTreeNode
    nodes: Mapping[UUID, _RefinementTreeNode[SuiteParameters, CaseParameters]]
    sample_ratio: float
    performance_drop_threshold: float
    guidelines: str | None = None
    rounds_remaining: int

    def leafs(self) -> Sequence[_RefinementTreeNode[SuiteParameters, CaseParameters]]:
        return [node for node in self.nodes.values() if node.is_leaf and not node.pruned]


def _select_focused_cases[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
](
    *,
    evaluation_result: SuiteEvaluatorResult[SuiteParameters, CaseParameters],
    evaluation_cases: Sequence[EvaluationSuiteCase[CaseParameters]],
    sample_ratio: float,
) -> Sequence[EvaluationSuiteCase[CaseParameters]]:
    # Get all failing cases
    failing_cases: list[EvaluationSuiteCase[CaseParameters]] = [
        case_result.case for case_result in evaluation_result.cases if not case_result.passed
    ]

    # Get passing cases and sample
    passing_cases: list[EvaluationSuiteCase[CaseParameters]] = [
        case_result.case for case_result in evaluation_result.cases if case_result.passed
    ]

    # Get other, previously excluded cases
    additional_cases: list[EvaluationSuiteCase[CaseParameters]] = [
        case for case in evaluation_cases if case not in evaluation_result.cases
    ]

    # Intelligent sampling: sample some passing cases
    sampling_cases_pool: list[EvaluationSuiteCase[CaseParameters]] = (
        passing_cases + additional_cases
    )
    sample_size: int = (
        max(1, int(len(sampling_cases_pool) * sample_ratio)) if sampling_cases_pool else 0
    )
    sampling_cases: list[EvaluationSuiteCase[CaseParameters]] = (
        random.sample(sampling_cases_pool, min(len(sampling_cases_pool), sample_size))  # nosec: B311
        if sample_size > 0
        else []
    )

    ctx.log_info(
        f"Created focused evaluation suite: "
        f"{len(failing_cases)} failing + {len(sampling_cases)} sampled other "
        f"(out of {len(passing_cases)} total passing)"
    )

    # Combine for focused evaluation
    return failing_cases + sampling_cases


def _tree_initialization_stage[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
    Value,
](
    *,
    instruction: Instruction,
    evaluation_suite: EvaluationSuite[SuiteParameters, CaseParameters, Value],
    sample_ratio: float,
    performance_drop_threshold: float,
    guidelines: str | None,
    rounds_limit: int,
) -> Stage:
    @stage
    async def tree_initialization(
        *,
        state: StageState,
    ) -> StageState:
        ctx.log_info("...evaluating initial instruction with full suite...")
        # Make initial evaluation
        evaluation: SuiteEvaluatorResult[SuiteParameters, CaseParameters]
        with ctx.updated(_instructions(instruction=instruction)):
            evaluation = await evaluation_suite()

        ctx.log_info(f"...initial score: {evaluation.relative_score:.4f}...")

        # Create root node
        root_node = _RefinementTreeNode(
            identifier=uuid4(),
            instruction=instruction,
            strategy="initial",
            parent_id=None,
            depth=0,
            children=(),
            # for root complete evaluation is the same as focused
            focused_evaluation=evaluation,
            complete_evaluation=evaluation,
            pruned=False,
        )

        # Setup initial state
        return state.updated(
            _RefinementState[SuiteParameters, CaseParameters](
                root=root_node,
                nodes={root_node.identifier: root_node},
                sample_ratio=sample_ratio,
                performance_drop_threshold=performance_drop_threshold,
                guidelines=guidelines,
                rounds_remaining=rounds_limit,
            ),
            # set the result to be initial instruction
            result=MultimodalContent.of(instruction.content),
        )

    return tree_initialization


def _tree_exploration_stage[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
    Value,
](
    *,
    evaluation_suite: EvaluationSuite[SuiteParameters, CaseParameters, Value],
    rounds_limit: int,
    quality_threshold: float,
) -> Stage:
    @stage
    async def tree_exploration(
        *,
        state: StageState,
    ) -> StageState:
        refinement_state: _RefinementState = state.get(
            _RefinementState,
            required=True,
        )
        evaluation_cases: Sequence[
            EvaluationSuiteCase[CaseParameters]
        ] = await evaluation_suite.cases()

        node_updates: Sequence[
            Mapping[UUID, _RefinementTreeNode[SuiteParameters, CaseParameters]]
        ] = await gather(
            *[
                _explore_node(
                    node=leaf,
                    evaluation_suite=evaluation_suite,
                    evaluation_cases=evaluation_cases,
                    sample_ratio=refinement_state.sample_ratio,
                    performance_drop_threshold=refinement_state.performance_drop_threshold,
                    guidelines=refinement_state.guidelines,
                )
                for leaf in refinement_state.leafs()
            ]
        )

        updated_nodes: dict[UUID, _RefinementTreeNode[SuiteParameters, CaseParameters]] = as_dict(
            refinement_state.nodes
        )
        for nodes in node_updates:
            updated_nodes = {
                **updated_nodes,
                **nodes,
            }

        # Update the refinement state with the updated nodes
        refinement_state = refinement_state.updated(nodes=updated_nodes)

        # Log tree statistics
        total_nodes: int = len(refinement_state.nodes)
        pruned_count: int = len([node for node in refinement_state.nodes.values() if node.pruned])
        active_nodes: int = total_nodes - pruned_count

        ctx.log_info(
            f"Tree exploration round complete. Nodes: {total_nodes} total, "
            f"{active_nodes} active, {pruned_count} pruned"
        )

        return state.updated(refinement_state)

    async def condition(
        *,
        state: StageState,
        iteration: int,
    ) -> bool:
        refinement_state: _RefinementState = state.get(
            _RefinementState,
            required=True,
        )
        if any(
            (
                node.complete_evaluation_score
                if node.complete_evaluation_score is not None
                else node.focused_evaluation_score
            )
            > quality_threshold
            for node in refinement_state.nodes.values()
        ):
            # we can complete early with exceptional score
            return False

        return iteration < rounds_limit

    return Stage.loop(
        tree_exploration,
        condition=condition,
    )


async def _explore_node[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
    Value,
](
    node: _RefinementTreeNode[SuiteParameters, CaseParameters],
    evaluation_suite: EvaluationSuite[SuiteParameters, CaseParameters, Value],
    evaluation_cases: Sequence[EvaluationSuiteCase[CaseParameters]],
    sample_ratio: float,
    performance_drop_threshold: float,
    guidelines: str | None,
) -> Mapping[UUID, _RefinementTreeNode[SuiteParameters, CaseParameters]]:
    assert not node.pruned  # nosec: B101 # skip pruned branches
    ctx.log_info(
        f"Exploring node {node.identifier} at"
        f" depth {node.depth}, score: {node.focused_evaluation_score:.4f}"
    )

    # Generate exactly 2 complementary strategies
    strategies: Sequence[tuple[str, Instruction]] = await _generate_refinement_strategies(
        instruction=node.instruction,
        evaluation_result=node.focused_evaluation,
        parent_strategy=node.strategy,
        guidelines=guidelines,
    )

    # Create and evaluate child nodes
    focused_suite_cases: Sequence[EvaluationSuiteCase[CaseParameters]] = _select_focused_cases(
        evaluation_result=node.focused_evaluation,
        evaluation_cases=evaluation_cases,
        sample_ratio=sample_ratio,
    )

    children: dict[UUID, _RefinementTreeNode[SuiteParameters, CaseParameters]] = {}
    for strategy_name, refined_instruction in strategies:
        # Evaluate with focused suite
        ctx.log_info(f"Evaluating strategy '{strategy_name}'...")
        focused_evaluation: SuiteEvaluatorResult[SuiteParameters, CaseParameters]
        with ctx.updated(_instructions(instruction=refined_instruction)):
            focused_evaluation = await evaluation_suite(*focused_suite_cases)

        # Check for performance drop
        performance_ratio = (
            focused_evaluation.relative_score / node.focused_evaluation_score
            if node.focused_evaluation_score > 0
            else 0
        )
        pruned: bool = performance_ratio < performance_drop_threshold

        # Create child node
        child_node = _RefinementTreeNode(
            identifier=uuid4(),
            instruction=refined_instruction,
            strategy=strategy_name,
            parent_id=node.identifier,
            depth=node.depth + 1,
            focused_evaluation=focused_evaluation,
            complete_evaluation=None,
            children=(),
            pruned=pruned,
        )
        children[child_node.identifier] = child_node

        if pruned:
            ctx.log_warning(
                f"Strategy '{strategy_name}' caused {(1 - performance_ratio) * 100:.1f}% "
                f"performance drop. Pruning this branch."
            )

        else:
            ctx.log_info(
                f"Created child node {child_node.identifier}: strategy={strategy_name}, "
                f"score={child_node.focused_evaluation_score:.4f}"
                f" ({performance_ratio:.1%} of parent)"
            )

    return children


# TODO: we should split instruction refinement from strategy selection
async def _generate_refinement_strategies[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
](
    instruction: Instruction,
    evaluation_result: SuiteEvaluatorResult[SuiteParameters, CaseParameters],
    parent_strategy: str | None,
    guidelines: str | None,
) -> Sequence[tuple[str, Instruction]]:
    # Analyze failures
    failure_report = evaluation_result.report(
        include_passed=False,
        include_details=True,
    )

    strategy_prompt: str = f"""
Based on the evaluation failures analysis, generate EXACTLY 2 DIFFERENT refinement strategies.

Current instruction: {instruction.content}
Parent strategy: {parent_strategy or "Initial"}

Generate 2 strategies that:
1. Take CONTRASTING approaches to fixing the issues
2. Address different aspects of the failures
3. Are mutually exclusive in their approach

Examples of contrasting approaches:
- Strategy 1: Add more specific details and examples
- Strategy 2: Simplify and remove potentially confusing elements

OR:
- Strategy 1: Add explicit constraints and rules
- Strategy 2: Make instructions more flexible and principle-based

{f"Additional guidelines: {guidelines}" if guidelines else ""}

For each strategy, provide:
<STRATEGY>
<NAME>[Brief descriptive name]</NAME>
<APPROACH>[One sentence explaining the approach]</APPROACH>
<CONTENT>[The refined instruction implementing this strategy]</CONTENT>
</STRATEGY>
"""

    response = await Stage.completion(
        f"Failure Analysis:\n{failure_report}",
        instruction=strategy_prompt,
    ).execute()

    # Parse strategies
    strategies: list[tuple[str, Instruction]] = []
    for strategy_element in MultimodalTagElement.parse("STRATEGY", content=response):
        name_elem: MultimodalTagElement | None = MultimodalTagElement.parse_first(
            "NAME", content=strategy_element.content
        )
        content_elem: MultimodalTagElement | None = MultimodalTagElement.parse_first(
            "CONTENT", content=strategy_element.content
        )

        if name_elem and content_elem:
            strategy_name: str = name_elem.content.to_str().strip()
            refined_content: str = content_elem.content.to_str().strip()
            refined_instruction: Instruction = instruction.updated(content=refined_content)
            strategies.append((strategy_name, refined_instruction))

    match len(strategies):
        case 2:
            return strategies

        case 1:
            ctx.log_warning(f"Expected 2 strategies, got {len(strategies)}")
            return strategies

        case 0:
            ctx.log_warning("Failed to generate refinement strategies")
            return strategies  # this branch will skipped

        case _:  # more than 2
            ctx.log_warning(f"Expected 2 strategies, got {len(strategies)}")
            return strategies[:2]


def _tree_finalization_stage[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
    Value,
](
    *,
    evaluation_suite: EvaluationSuite[SuiteParameters, CaseParameters, Value],
    quality_threshold: float,
    candidates_limit: int,
) -> Stage:
    @stage
    async def tree_finalization(
        *,
        state: StageState,
    ) -> StageState:
        refinement_state: _RefinementState = state.get(
            _RefinementState,
            required=True,
        )
        # Find best candidates across entire tree (except root)
        candidates: Sequence[_RefinementTreeNode] = _find_best_candidates(
            refinement_state=refinement_state,
            quality_threshold=quality_threshold,
            limit=candidates_limit,
        )

        if not candidates:
            ctx.log_warning("No candidates found, keeping original instruction")
            return state

        ctx.log_info(f"Running full evaluation on top {len(candidates)} candidates")

        # Full evaluation of top candidates
        best_instruction: Instruction = refinement_state.root.instruction
        best_score: float = refinement_state.root.complete_evaluation_score or 0
        best_node: _RefinementTreeNode = refinement_state.root

        for candidate_node in candidates:
            ctx.log_info(
                f"Full evaluation of node {candidate_node.identifier} "
                f"(strategy: {candidate_node.strategy}, depth: {candidate_node.depth})"
            )

            complete_evaluation: SuiteEvaluatorResult[SuiteParameters, CaseParameters]
            with ctx.updated(_instructions(instruction=candidate_node.instruction)):
                complete_evaluation = await evaluation_suite()

            # Update node with full eval score
            updated_node: _RefinementTreeNode = candidate_node.updated(
                complete_evaluation=complete_evaluation
            )
            refinement_state = refinement_state.updated(
                nodes={
                    **refinement_state.nodes,
                    # update node in tree
                    updated_node.identifier: updated_node,
                },
            )

            ctx.log_info(
                f"Full evaluation score: {complete_evaluation.relative_score:.4f} "
                f"(focused was: {updated_node.focused_evaluation_score:.4f})"
            )

            if complete_evaluation.relative_score > best_score:
                best_instruction = updated_node.instruction
                best_score = complete_evaluation.relative_score
                best_node = updated_node

        # Log final statistics
        _log_tree_statistics(
            refinement_state=refinement_state,
            best_node=best_node,
        )

        # Update result with best instruction and updated state
        return state.updated(
            refinement_state,
            result=MultimodalContent.of(best_instruction.content),
        )

    return tree_finalization


def _find_best_candidates(
    *,
    refinement_state: _RefinementState,
    quality_threshold: float,
    limit: int,
) -> Sequence[_RefinementTreeNode]:
    # Get all nodes that are not pruned
    candidates: list[_RefinementTreeNode] = []
    for node in refinement_state.nodes.values():
        # skip excluded and root from candidates
        if node.pruned or node.is_root:
            continue

        # Include if it's a leaf or has exceptional performance
        if node.is_leaf or node.focused_evaluation_score > quality_threshold:
            candidates.append(node)

    # Sort by focused score and depth (prefer deeper nodes with same score)
    candidates.sort(
        key=lambda x: (x.focused_evaluation_score, x.depth),
        reverse=True,
    )

    return candidates[:limit]


def _log_tree_statistics(
    *,
    refinement_state: _RefinementState,
    best_node: _RefinementTreeNode,
) -> None:
    total_nodes: int = len(refinement_state.nodes)
    pruned_count: int = len([node for node in refinement_state.nodes.values() if node.pruned])
    active_nodes: int = total_nodes - pruned_count

    # Depth distribution
    depth_distribution: dict[int, int] = {}
    for node in refinement_state.nodes.values():
        depth_distribution[node.depth] = depth_distribution.get(node.depth, 0) + 1

    max_depth: int = max(depth_distribution.keys()) if depth_distribution else 0
    theoretical_max: int = 2 ** (max_depth + 1) - 1
    efficiency: float = active_nodes / theoretical_max if theoretical_max > 0.0 else 0.0

    # Build path to best node
    best_path: list[str] = []
    current_id: UUID | None = best_node.identifier
    while current_id:
        node = refinement_state.nodes[current_id]
        best_path.append(f"{node.strategy} (score: {node.focused_evaluation_score:.3f})")
        current_id = node.parent_id

    best_path.reverse()

    ctx.log_info(
        f"Tree statistics:\n"
        f"- Total nodes explored: {total_nodes}\n"
        f"- Active nodes: {active_nodes}\n"
        f"- Pruned nodes: {pruned_count} ({pruned_count / total_nodes * 100:.1f}%)\n"
        f"- Max depth reached: {max_depth}\n"
        f"- Exploration efficiency: {efficiency:.1%}\n"
        f"- Best path: {' -> '.join(best_path) if best_path else 'None'}"
    )
