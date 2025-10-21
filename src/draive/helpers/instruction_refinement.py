import random
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID, uuid4

from haiway import Meta, State, as_dict, ctx, execute_concurrently

from draive.evaluation import (
    EvaluatorSuiteCase,
    EvaluatorSuiteResult,
    PreparedEvaluatorSuite,
)
from draive.models import ModelInstructions
from draive.multimodal import (
    MultimodalContent,
    MultimodalTag,
    Template,
    TemplatesRepository,
    TextContent,
)
from draive.parameters import DataModel
from draive.stages import Stage, StageState, stage

__all__ = ("refine_instructions",)


async def refine_instructions[Parameters: DataModel](
    instructions: Template,
    /,
    *,
    instructions_content: ModelInstructions | None = None,
    guidelines: str | None = None,
    evaluator_suite: PreparedEvaluatorSuite[Parameters],
    evaluator_cases: Sequence[EvaluatorSuiteCase[Parameters]],
    rounds_limit: int,
    sample_ratio: float = 0.1,
    candidates_limit: int = 3,
    performance_drop_threshold: float = 0.5,
    quality_threshold: float = 0.99,
    concurrent_nodes: int = 1,
) -> ModelInstructions:
    """
    Refine instructions using binary tree exploration with performance pruning.

    Each node generates exactly 2 complementary strategies.
    Branches are pruned if performance drops below threshold.

    Args:
        instructions: The instructions to refine
        guidelines: Optional guidelines for refinement
        evaluator_suite: Suite to evaluate instructions performance
        rounds_limit: Maximum depth of refinement tree
        sample_ratio: Fraction of passing cases to include in focused evaluation
        candidates_limit: Number of top candidates to fully evaluate
        performance_drop_threshold: Prune branches with score drop > this threshold
        quality_threshold: Stop if score reaches this threshold
        concurrent_nodes: How many nodes explored concurrently
    """

    assert rounds_limit > 0  # nosec: B101
    assert 1 >= sample_ratio > 0  # nosec: B101
    assert candidates_limit > 0  # nosec: B101
    assert 1 >= performance_drop_threshold > 0  # nosec: B101
    assert 1 >= quality_threshold >= 0  # nosec: B101
    assert concurrent_nodes > 0  # nosec: B101
    assert len(evaluator_cases) > 0  # nosec: B101

    ctx.log_info(
        f"Starting tree-based refinement: "
        f"max_depth={rounds_limit}, "
        f"sample_ratio={sample_ratio}, "
        f"performance_drop_threshold={performance_drop_threshold}"
    )

    result: MultimodalContent = await Stage.sequence(
        _tree_initialization_stage(
            instructions=instructions,
            instructions_content=instructions_content,
            evaluator_suite=evaluator_suite,
            sample_ratio=sample_ratio,
            performance_drop_threshold=performance_drop_threshold,
            guidelines=guidelines,
            rounds_limit=rounds_limit,
        ),
        _tree_exploration_stage(
            evaluator_suite=evaluator_suite,
            evaluator_cases=evaluator_cases,
            rounds_limit=rounds_limit,
            quality_threshold=quality_threshold,
            concurrent_nodes=concurrent_nodes,
        ),
        _tree_finalization_stage(
            evaluator_suite=evaluator_suite,
            quality_threshold=quality_threshold,
            candidates_limit=candidates_limit,
        ),
    ).execute()

    ctx.log_info("...instructions refinement finished!")
    return result.to_str()


class _RefinementTreeNode(State):
    identifier: UUID
    instructions: Template
    instructions_content: ModelInstructions
    strategy: str
    parent_id: UUID | None
    depth: int
    focused_evaluation: EvaluatorSuiteResult
    complete_evaluation: EvaluatorSuiteResult | None
    children: Sequence[UUID]
    pruned: bool

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def focused_evaluation_performance(self) -> float:
        return self.focused_evaluation.performance

    @property
    def complete_evaluation_performance(self) -> float | None:
        if self.complete_evaluation is None:
            return None

        return self.complete_evaluation.performance

    @property
    def patched_instructions_repository(self) -> TemplatesRepository:
        repository: TemplatesRepository = ctx.state(TemplatesRepository)

        async def instructions_loading(
            identifier: str,
            meta: Meta,
            **extra: Any,
        ) -> str | None:
            # use updated instructions if able
            if identifier == self.instructions.identifier:
                return self.instructions_content

            else:  # otherwise preserve unchanged instructions
                return await repository.loading(
                    identifier=identifier,
                    meta=meta,
                    **extra,
                )

        return TemplatesRepository(
            listing=repository.listing,  # keep current listing
            loading=instructions_loading,  # replace loading
            # do not allow modifications - use default noop implementation
        )


class _RefinementState(State):
    root: _RefinementTreeNode
    nodes: Mapping[UUID, _RefinementTreeNode]
    sample_ratio: float
    performance_drop_threshold: float
    guidelines: str | None = None
    rounds_remaining: int

    def leaves(self) -> Sequence[_RefinementTreeNode]:
        return [node for node in self.nodes.values() if node.is_leaf and not node.pruned]


def _select_focused_cases[Parameters: DataModel](
    *,
    evaluation_result: EvaluatorSuiteResult,
    evaluator_cases: Sequence[EvaluatorSuiteCase[Parameters]],
    sample_ratio: float,
) -> Sequence[str]:
    # Get all failing cases
    failing_cases: Sequence[str] = [
        case_result.case_identifier
        for case_result in evaluation_result.results
        if not case_result.passed
    ]

    # Get passing cases
    passing_cases: Sequence[str] = [
        case_result.case_identifier
        for case_result in evaluation_result.results
        if case_result.passed
    ]

    # Get other, previously excluded cases
    additional_cases: Sequence[str] = [
        case.identifier
        for case in evaluator_cases
        if case.identifier not in failing_cases and case.identifier not in passing_cases
    ]

    sampling_cases_pool: Sequence[str] = passing_cases + additional_cases
    sample_size: int = (
        max(1, int(len(sampling_cases_pool) * sample_ratio)) if sampling_cases_pool else 0
    )
    sampling_cases: Sequence[str] = (
        random.sample(  # nosec: B311
            sampling_cases_pool,
            min(len(sampling_cases_pool), sample_size),
        )
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


def _tree_initialization_stage[Parameters: DataModel](
    *,
    instructions: Template,
    instructions_content: ModelInstructions | None,
    evaluator_suite: PreparedEvaluatorSuite[Parameters],
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
        ctx.log_info("...evaluating initial instructions with full suite...")
        # Make initial evaluation using current instructions
        evaluation: EvaluatorSuiteResult = await evaluator_suite()

        ctx.log_info(f"...initial score: {evaluation.performance:.2f}...")

        content: ModelInstructions
        if instructions_content is None:
            content = await TemplatesRepository.load(instructions)

        else:
            content = instructions_content

        # Create root node
        root_node = _RefinementTreeNode(
            identifier=uuid4(),
            instructions=instructions,
            instructions_content=content,
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
            _RefinementState(
                root=root_node,
                nodes={root_node.identifier: root_node},
                sample_ratio=sample_ratio,
                performance_drop_threshold=performance_drop_threshold,
                guidelines=guidelines,
                rounds_remaining=rounds_limit,
            ),
            # set the result to be initial instruction content
            result=MultimodalContent.of(content),
        )

    return tree_initialization


def _tree_exploration_stage[Parameters: DataModel](
    *,
    evaluator_suite: PreparedEvaluatorSuite[Parameters],
    evaluator_cases: Sequence[EvaluatorSuiteCase[Parameters]],
    rounds_limit: int,
    quality_threshold: float,
    concurrent_nodes: int,
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

        async def explore(
            node: _RefinementTreeNode,
        ) -> Mapping[UUID, _RefinementTreeNode]:
            return await _explore_node(
                node=node,
                evaluator_suite=evaluator_suite,
                evaluator_cases=evaluator_cases,
                sample_ratio=refinement_state.sample_ratio,
                performance_drop_threshold=refinement_state.performance_drop_threshold,
                guidelines=refinement_state.guidelines,
            )

        node_updates: Sequence[Mapping[UUID, _RefinementTreeNode]] = await execute_concurrently(
            explore,
            refinement_state.leaves(),
            concurrent_tasks=concurrent_nodes,
        )

        updated_nodes: dict[UUID, _RefinementTreeNode] = as_dict(refinement_state.nodes)
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
                node.complete_evaluation_performance
                if node.complete_evaluation_performance is not None
                else node.focused_evaluation_performance
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


async def _explore_node[Parameters: DataModel](
    node: _RefinementTreeNode,
    evaluator_suite: PreparedEvaluatorSuite[Parameters],
    evaluator_cases: Sequence[EvaluatorSuiteCase[Parameters]],
    sample_ratio: float,
    performance_drop_threshold: float,
    guidelines: str | None,
) -> Mapping[UUID, _RefinementTreeNode]:
    assert not node.pruned  # nosec: B101 # skip pruned branches
    ctx.log_info(
        f"Exploring node {node.identifier} at"
        f" depth {node.depth}, score: {node.focused_evaluation_performance:.2f}"
    )

    # Generate exactly 2 complementary strategies
    strategies: Sequence[tuple[str, ModelInstructions]] = await _generate_refined_instructions(
        current_content=node.instructions_content,
        evaluation_result=node.focused_evaluation,
        parent_strategy=node.strategy,
        guidelines=guidelines,
    )

    # Create and evaluate child nodes
    focused_suite_cases: Sequence[str] = _select_focused_cases(
        evaluation_result=node.focused_evaluation,
        evaluator_cases=evaluator_cases,
        sample_ratio=sample_ratio,
    )

    children: dict[UUID, _RefinementTreeNode] = {}
    for strategy_name, refined_instructions in strategies:
        # Create child node
        child_node = _RefinementTreeNode(
            identifier=uuid4(),
            instructions=node.instructions,
            instructions_content=refined_instructions,
            strategy=strategy_name,
            parent_id=node.identifier,
            depth=node.depth + 1,
            # initialize with placeholder
            focused_evaluation=EvaluatorSuiteResult(
                suite="placeholder",
                results=(),
            ),
            complete_evaluation=None,
            children=(),
            pruned=False,
        )

        # Evaluate with focused suite
        ctx.log_info(f"Evaluating strategy '{strategy_name}'...")
        focused_evaluation: EvaluatorSuiteResult
        with ctx.updated(child_node.patched_instructions_repository):
            focused_evaluation = await evaluator_suite(focused_suite_cases)

        # Check for performance drop
        performance_ratio = (
            focused_evaluation.performance / node.focused_evaluation_performance
            if node.focused_evaluation_performance > 0
            else 0
        )
        pruned: bool = performance_ratio < performance_drop_threshold

        # update node with evaluation data
        children[child_node.identifier] = child_node.updated(
            focused_evaluation=focused_evaluation,
            pruned=pruned,
        )

        if pruned:
            ctx.log_warning(
                f"Strategy '{strategy_name}' caused {(1 - performance_ratio) * 100:.1f}% "
                f"performance drop. Pruning this branch."
            )

        else:
            ctx.log_info(
                f"Created child node {child_node.identifier}: strategy={strategy_name}, "
                f"score={focused_evaluation.performance:.2f}"
                f" ({performance_ratio:.1%} of parent)"
            )

    return children


async def _generate_strategy_metadata(
    *,
    evaluation_result: EvaluatorSuiteResult,
    parent_strategy: str | None,
    guidelines: str | None,
) -> Sequence[tuple[str, str]]:
    """
    Generate strategy metadata (name and approach) without instructions content.

    Returns:
        Sequence of tuples containing (name, approach) for each strategy.
    """
    # Analyze failures
    failure_report = evaluation_result.report(
        detailed=True,
        include_passed=False,
    )

    strategy_prompt: str = f"""
You are an expert prompt engineer providing feedback and recommendations for instructions improvement.
Based on the evaluation failures analysis, prepare EXACTLY 2 DIFFERENT refinement strategies.

<previous_strategy>{parent_strategy or "Initial"}</previous_strategy>
<task>
Generate 2 instructions refinement strategies and recommendations that:
1. Take CONTRASTING approaches to fixing the issues
2. Address different aspects of the failures
3. Are mutually exclusive in their approach
</task>
{f"\n<guidelines>\n{guidelines}</guidelines>\n" if guidelines else ""}
<examples>
<strategy>
<name>Exemplification</name>
<approach>Add more specific details and examples</approach>
</strategy>
<strategy>
<name>Constraining</name>
<approach>Add explicit constraints and rules</approach>
</strategy>
</examples>
<format>
For each strategy, provide the name and approach in the following tags format:
<strategy>
<name>[Brief descriptive name]</name>
<approach>[Explanation of the approach with description of recommendations]</approach>
</strategy>
</format>
"""  # noqa: E501

    response: MultimodalContent = await Stage.completion(
        f"<failure_analysis>\n{failure_report}\n</failure_analysis>",
        instructions=strategy_prompt,
    ).execute()

    # Parse strategies
    strategies: list[tuple[str, str]] = []
    for strategy_element in response.tags("strategy"):
        name_element: MultimodalTag | None = strategy_element.content.tag("name")
        approach_element: MultimodalTag | None = strategy_element.content.tag("approach")

        if name_element and approach_element:
            strategy_name: str = name_element.content.to_str().strip()
            strategy_approach: str = approach_element.content.to_str().strip()
            strategies.append((strategy_name, strategy_approach))

    return strategies


async def _generate_instructions(
    *,
    current_content: ModelInstructions,
    strategy_name: str,
    strategy_approach: str,
    evaluation_result: EvaluatorSuiteResult,
    guidelines: str | None,
) -> ModelInstructions:
    # Analyze failures
    failure_report: str = evaluation_result.report(
        detailed=True,
        include_passed=False,
    )

    refinement_prompt: str = f"""\
You are an expert prompt engineer refining provided instructions referred as a subject.
Provide an updated version of the subject that implements the described strategy to address the evaluation failures.
Strictly focus on delivering the best possible instructions content with excellent clarity and performance.
Ensure to apply modifications by following requested strategy while keeping original instructions intent, goal and formatting style.

<strategy>
<name>
{strategy_name}
</name>
<approach>
{strategy_approach}
</approach>
</strategy>

<subject>
{current_content}
</subject>
{f"\n<guidelines>\n{guidelines}</guidelines>\n" if guidelines else ""}
<format>
Provide ONLY the refined instructions content, without any explanation or metadata.
</format>
"""  # noqa: E501

    ctx.log_info(f"Generating updated instructions using {strategy_name}:\n\n{strategy_approach}")
    response: MultimodalContent = await Stage.completion(
        f"<failure_analysis>\n{failure_report}\n</failure_analysis>",
        instructions=refinement_prompt,
    ).execute()

    updated_instruction: str = response.to_str().strip()

    ctx.log_info(f"Prepared updated instructions using {strategy_name}")
    ctx.log_debug(f"Updated instruction for {strategy_name}: {updated_instruction}")

    return updated_instruction


async def _generate_refined_instructions(
    *,
    current_content: ModelInstructions,
    evaluation_result: EvaluatorSuiteResult,
    parent_strategy: str | None,
    guidelines: str | None,
) -> Sequence[tuple[str, ModelInstructions]]:
    # Step 1: Generate strategy metadata
    strategy_metadata: Sequence[tuple[str, str]] = await _generate_strategy_metadata(
        evaluation_result=evaluation_result,
        parent_strategy=parent_strategy,
        guidelines=guidelines,
    )

    # Step 2: Generate instructions content for each strategy
    refined_strategies: list[tuple[str, ModelInstructions]] = []
    for strategy_name, strategy_approach in strategy_metadata:
        refined_instructions = await _generate_instructions(
            current_content=current_content,
            strategy_name=strategy_name,
            strategy_approach=strategy_approach,
            evaluation_result=evaluation_result,
            guidelines=guidelines,
        )
        refined_strategies.append((strategy_name, refined_instructions))

    # Validate and return results
    match len(refined_strategies):
        case 2:
            return refined_strategies

        case 1:
            ctx.log_warning(f"Expected 2 strategies, got {len(refined_strategies)}")
            return refined_strategies

        case 0:
            ctx.log_warning("Failed to generate refinement strategies")
            return refined_strategies  # this branch will be skipped

        case _:  # more than 2
            ctx.log_warning(f"Expected 2 strategies, got {len(refined_strategies)}")
            return refined_strategies[:2]


def _tree_finalization_stage[Parameters: DataModel](
    *,
    evaluator_suite: PreparedEvaluatorSuite[Parameters],
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
        best_instructions: ModelInstructions = refinement_state.root.instructions_content
        best_score: float = refinement_state.root.complete_evaluation_performance or 0
        best_node: _RefinementTreeNode = refinement_state.root

        for candidate_node in candidates:
            ctx.log_info(
                f"Full evaluation of node {candidate_node.identifier} "
                f"(strategy: {candidate_node.strategy}, depth: {candidate_node.depth})"
            )

            complete_evaluation: EvaluatorSuiteResult
            with ctx.updated(candidate_node.patched_instructions_repository):
                complete_evaluation = await evaluator_suite()

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
                f"Full evaluation score: {complete_evaluation.performance:.4f} "
                f"(focused was: {updated_node.focused_evaluation_performance:.4f})"
            )

            if complete_evaluation.performance > best_score:
                best_instructions = updated_node.instructions_content
                best_score = complete_evaluation.performance
                best_node = updated_node

        # Log final statistics
        _log_tree_statistics(
            refinement_state=refinement_state,
            best_node=best_node,
        )

        # Update result with best instructions and updated state
        return state.updated(
            refinement_state,
            result=MultimodalContent.of(
                TextContent.of(
                    best_instructions,
                    meta={"score": best_score},
                ),
            ),
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
        if node.is_leaf or node.focused_evaluation_performance > quality_threshold:
            candidates.append(node)

    # Sort by focused score and depth (prefer deeper nodes with same score)
    candidates.sort(
        key=lambda x: (x.focused_evaluation_performance, x.depth),
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

    # Build path to best node
    best_path: list[str] = []
    current_id: UUID | None = best_node.identifier
    while current_id:
        node = refinement_state.nodes[current_id]
        best_path.append(f"{node.strategy} (score: {node.focused_evaluation_performance:.3f})")
        current_id = node.parent_id

    best_path.reverse()

    ctx.log_info(
        f"Tree statistics:\n"
        f"- Total nodes explored: {total_nodes}\n"
        f"- Active nodes: {active_nodes}\n"
        f"- Pruned nodes: {pruned_count} ({pruned_count / total_nodes * 100:.1f}%)\n"
        f"- Max depth reached: {max_depth}\n"
        f"- Best path: {' -> '.join(best_path) if best_path else 'None'}"
    )
