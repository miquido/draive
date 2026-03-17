from collections.abc import AsyncIterable, Iterable, Sequence
from typing import Any

import pytest
from haiway import State

from draive.evaluation import EvaluatorResult
from draive.models import (
    GenerativeModel,
    ModelContextElement,
    ModelInput,
    ModelOutput,
    ModelOutputChunk,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
    ModelTools,
)
from draive.multimodal import MultimodalContent, TextContent
from draive.steps import Step, StepException, StepState, step
from draive.tools import tool


class AlphaArtifact(State):
    value: str


class BetaArtifact(State):
    value: str


class ToggleArtifact(State):
    value: str


def _text_of(content: MultimodalContent) -> str:
    return "".join(part.text for part in content.parts if isinstance(part, TextContent))


async def _stream_of(*chunks: ModelOutputChunk) -> AsyncIterable[ModelOutputChunk]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_emitting_stream_and_run_collects_content() -> None:
    step_obj = Step.emitting(TextContent.of("a"), TextContent.of("b"))

    streamed = [part async for part in step_obj.stream()]
    result = await step_obj.run()

    assert [part.text for part in streamed if isinstance(part, TextContent)] == ["a", "b"]
    assert _text_of(result) == "ab"


@pytest.mark.asyncio
async def test_updating_artifacts_and_get() -> None:
    state = await Step.updating_artifacts(
        AlphaArtifact(value="alpha"),
        custom=BetaArtifact(value="beta"),
    ).process()

    assert state.get(AlphaArtifact, required=True).value == "alpha"
    assert state.get(BetaArtifact, key="custom", required=True).value == "beta"


@pytest.mark.asyncio
async def test_appending_and_updating_context() -> None:
    initial_context = (ModelInput.of(MultimodalContent.of("input")),)

    async def reverse_context(
        context: Sequence[ModelContextElement],
    ) -> Sequence[ModelContextElement]:
        return tuple(reversed(context))

    pipeline = Step.sequence(
        Step.appending_context(ModelOutput.of(MultimodalContent.of("output"))),
        Step.updating_context(reverse_context),
    )

    state = await pipeline.process(initial_context)

    assert len(state.context) == 2
    assert isinstance(state.context[0], ModelOutput)
    assert isinstance(state.context[1], ModelInput)


@pytest.mark.asyncio
async def test_preserving_and_restoring_state() -> None:
    storage: dict[str, StepState] = {"state": StepState.of()}

    async def preserving(state: StepState) -> None:
        storage["state"] = state

    async def restoring() -> StepState:
        return storage["state"]

    original = StepState.of(
        (ModelInput.of(MultimodalContent.of("saved")),),
        AlphaArtifact(value="stored"),
    )

    await Step.preserving_state(preserving).process(
        original.context,
        AlphaArtifact(value="stored"),
    )
    restored = await Step.restoring_state(restoring).process()

    assert restored.context == original.context
    assert restored.get(AlphaArtifact, required=True).value == "stored"


@pytest.mark.asyncio
async def test_restoring_state_with_snapshot_replaces_current_state() -> None:
    restored_snapshot = StepState.of(
        (ModelOutput.of(MultimodalContent.of("restored")),),
        BetaArtifact(value="restored"),
    )

    restored = await Step.restoring_state(restored_snapshot).process(
        (ModelInput.of(MultimodalContent.of("original")),),
        AlphaArtifact(value="original"),
    )

    assert restored is restored_snapshot
    assert restored.context == restored_snapshot.context
    assert restored.get(BetaArtifact, required=True).value == "restored"
    assert restored.get(AlphaArtifact) is None


@pytest.mark.asyncio
async def test_appending_input_and_output() -> None:
    async def input_provider() -> str:
        return "user"

    async def output_provider() -> str:
        return "assistant"

    pipeline = Step.sequence(
        Step.appending_input(input_provider),
        Step.appending_output(output_provider, emitting=True),
    )

    emitted = await pipeline.run()
    state = await pipeline.process()

    assert _text_of(emitted) == "assistant"
    assert len(state.context) == 2
    assert isinstance(state.context[0], ModelInput)
    assert isinstance(state.context[1], ModelOutput)


@pytest.mark.asyncio
async def test_loop_and_concurrent_composition() -> None:
    loop_step = Step.loop(
        Step.appending_context(ModelInput.of(MultimodalContent.of("tick"))),
        condition=lambda state, iteration: _bool_value(iteration < 2),
    )
    loop_state = await loop_step.process()
    assert len(loop_state.context) == 2

    async def merge_states(branches: Iterable[StepState]) -> StepState:
        left, right = tuple(branches)
        return left.updating(artifacts={**left.artifacts, **right.artifacts})

    concurrent_step = Step.concurrent(
        Step.updating_artifacts(left=AlphaArtifact(value="L")),
        Step.updating_artifacts(right=BetaArtifact(value="R")),
        merge=merge_states,
    )
    concurrent_state = await concurrent_step.process()

    assert concurrent_state.get(AlphaArtifact, key="left", required=True).value == "L"
    assert concurrent_state.get(BetaArtifact, key="right", required=True).value == "R"


@pytest.mark.asyncio
async def test_selection_executes_selected_step() -> None:
    beta = Step.updating_artifacts(BetaArtifact(value="B"))

    async def selecting(
        state: StepState,
    ) -> Step:
        _ = state
        return beta

    state = await Step.selection(selecting=selecting).process()

    assert state.get(BetaArtifact, required=True).value == "B"
    assert state.get(AlphaArtifact) is None


@pytest.mark.asyncio
async def test_selection_uses_current_state_to_choose_step() -> None:
    async def selecting(
        state: StepState,
    ) -> Step:
        if state.get(ToggleArtifact, key="toggle", required=True).value == "beta":
            return Step.updating_artifacts(BetaArtifact(value="B"))

        return Step.updating_artifacts(AlphaArtifact(value="A"))

    state = await Step.selection(selecting=selecting).process(
        toggle=ToggleArtifact(value="beta"),
    )

    assert state.get(BetaArtifact, required=True).value == "B"
    assert state.get(AlphaArtifact) is None


async def _bool_value(value: bool) -> bool:
    return value


@pytest.mark.asyncio
async def test_generating_completion_assembles_output_blocks() -> None:
    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        _ = (instructions, tools, context, output, extra)
        return _stream_of(
            TextContent.of("hello"),
            ModelReasoningChunk.of(TextContent.of("think")),
            ModelToolRequest.of("t1", tool="lookup", arguments={"q": "x"}),
            TextContent.of("done"),
        )

    completion = Step.generating_completion(
        instructions="test",
        tools=ModelTools.none,
        output="auto",
    ).with_ctx(GenerativeModel(generating=generating))

    state = await completion.process()

    assert len(state.context) == 1
    output = state.context[0]
    assert isinstance(output, ModelOutput)
    assert len(output.output) == 4
    assert isinstance(output.output[1], ModelReasoning)
    assert output.tool_requests[0].tool == "lookup"


@pytest.mark.asyncio
async def test_handling_tools_appends_tool_response_input() -> None:
    @tool
    async def echo(value: str) -> str:
        return f"E:{value}"

    request = ModelToolRequest.of("r1", tool="echo", arguments={"value": "x"})

    state = await Step.handling_tools([echo]).process((ModelOutput.of(request),))

    assert len(state.context) == 2
    assert isinstance(state.context[1], ModelInput)
    response = state.context[1].tool_responses[0]
    assert isinstance(response, ModelToolResponse)
    assert response.status == "success"
    assert _text_of(response.result) == "E:x"


@pytest.mark.asyncio
async def test_handling_tools_output_mode_appends_model_output() -> None:
    @tool(handling="output")
    async def amplify(value: str) -> str:
        return f"OUT:{value}"

    request = ModelToolRequest.of("r1", tool="amplify", arguments={"value": "x"})

    state = await Step.handling_tools([amplify]).process((ModelOutput.of(request),))

    assert len(state.context) == 3
    assert isinstance(state.context[1], ModelInput)
    assert isinstance(state.context[2], ModelOutput)
    assert _text_of(state.context[2].content) == "OUT:x"


@pytest.mark.asyncio
async def test_looping_completion_handles_tool_roundtrip() -> None:
    calls = 0

    @tool
    async def echo(value: str) -> str:
        return f"TOOL:{value}"

    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal calls
        _ = (instructions, tools, context, output, extra)
        calls += 1
        if calls == 1:
            return _stream_of(ModelToolRequest.of("c1", tool="echo", arguments={"value": "x"}))

        return _stream_of(TextContent.of("final"))

    looping = Step.looping_completion(
        instructions="loop",
        tools=[echo],
        output="auto",
    ).with_ctx(GenerativeModel(generating=generating))

    state = await looping.process()

    assert calls == 2
    assert isinstance(state.context[-1], ModelOutput)
    assert _text_of(state.context[-1].content) == "final"


@pytest.mark.asyncio
async def test_wrappers_retry_fallback_condition_and_suppressed_output() -> None:
    attempts = 0

    @step
    async def flaky(state: StepState) -> StepState:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("retry")

        return state.appending_context(ModelInput.of(MultimodalContent.of("ok")))

    @step
    async def fallback(state: StepState) -> StepState:
        return state.appending_context(ModelInput.of(MultimodalContent.of("fallback")))

    retried = flaky.with_retry(limit=2, catching=ValueError)
    retried_state = await retried.process()
    assert attempts == 2
    assert isinstance(retried_state.context[-1], ModelInput)

    @step
    async def always_fail(state: StepState) -> StepState:
        raise RuntimeError("boom")

    fallback_state = await always_fail.with_fallback(fallback, catching=RuntimeError).process()
    assert _text_of(fallback_state.context[-1].content) == "fallback"

    conditional = Step.emitting(TextContent.of("yes")).with_condition(
        False,
        alternative=Step.emitting(TextContent.of("no")),
    )
    assert _text_of(await conditional.run()) == "no"

    suppressed = Step.emitting(TextContent.of("x")).with_suppressed_output()
    assert _text_of(await suppressed.run()) == ""


@pytest.mark.asyncio
async def test_volatile_context_and_evaluation_wrappers() -> None:
    initial_context = (ModelInput.of(MultimodalContent.of("start")),)

    mutated = Step.appending_context(
        ModelOutput.of(MultimodalContent.of("temp")),
    ).with_volatile_context()
    volatile_state = await mutated.process(initial_context)
    assert volatile_state.context == initial_context

    async def failed_eval(_: Sequence[ModelContextElement]) -> EvaluatorResult:
        return EvaluatorResult.of("context", score=0.0, threshold=1.0)

    with pytest.raises(StepException):
        await Step.noop.with_context_evaluation(failed_eval, raise_on_failure=True).process()

    async def failed_output_eval(_: Sequence[Any]) -> EvaluatorResult:
        return EvaluatorResult.of("output", score=0.0, threshold=1.0)

    with pytest.raises(StepException):
        await (
            Step.emitting(TextContent.of("x"))
            .with_output_evaluation(
                failed_output_eval,
                raise_on_failure=True,
            )
            .run()
        )
