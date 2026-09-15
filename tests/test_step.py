import asyncio
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
async def test_emitting_normalizes_multimodal_values() -> None:
    # only content parts are recognized as emitted output, plain strings and
    # content containers have to be normalized instead of passed through
    step_obj = Step.emitting("a", MultimodalContent.of("b", "c"))

    streamed = [part async for part in step_obj.stream()]
    result = await step_obj.run()

    assert all(isinstance(part, TextContent) for part in streamed)
    assert _text_of(result) == "abc"


@pytest.mark.asyncio
async def test_emitting_empty_content_is_noop() -> None:
    assert await Step.emitting(MultimodalContent.empty).run() == MultimodalContent.empty


@pytest.mark.asyncio
async def test_updating_artifacts_and_get() -> None:
    state = await Step.updating_artifacts(
        AlphaArtifact(value="alpha"),
        custom=BetaArtifact(value="beta"),
    ).process()

    assert state.get(AlphaArtifact, required=True).value == "alpha"
    assert state.get(BetaArtifact, key="custom", required=True).value == "beta"


@pytest.mark.asyncio
async def test_appending_and_mutating_context() -> None:
    initial_context = (ModelInput.of(MultimodalContent.of("input")),)

    async def reverse_context(
        context: Sequence[ModelContextElement],
    ) -> Sequence[ModelContextElement]:
        return tuple(reversed(context))

    pipeline = Step.sequence(
        Step.appending_context(ModelOutput.of(MultimodalContent.of("output"))),
        Step.mutating_context(reverse_context),
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
    assert _text_of(response.content) == "E:x"


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


@pytest.mark.asyncio
async def test_volatile_tools_preserves_state_without_tool_context() -> None:
    state = (
        await Step.updating_artifacts(AlphaArtifact(value="kept")).with_volatile_tools().process()
    )

    assert state.get(AlphaArtifact, required=True).value == "kept"


@pytest.mark.asyncio
async def test_volatile_tools_only_strips_tools_from_added_context() -> None:
    existing = ModelOutput.of(
        MultimodalContent.of("existing"),
        ModelToolRequest.of("existing", tool="lookup", arguments={}),
    )
    added = ModelOutput.of(
        MultimodalContent.of("added"),
        ModelToolRequest.of("added", tool="lookup", arguments={}),
    )

    state = await Step.appending_context(added).with_volatile_tools().process((existing,))

    assert state.context[0] is existing
    assert state.context[0].contains_tools
    assert isinstance(state.context[1], ModelOutput)
    assert not state.context[1].contains_tools
    assert _text_of(state.context[1].content) == "added"


@pytest.mark.asyncio
async def test_context_replacement_accepts_async_callable_instance() -> None:
    replacement = (ModelInput.of(MultimodalContent.of("replacement")),)

    class ContextProvider:
        async def __call__(self) -> Sequence[ModelContextElement]:
            return replacement

    state = await Step.replacing_context(ContextProvider()).process()

    assert state.context == replacement


@pytest.mark.asyncio
async def test_isolated_context_accepts_async_callable_instance() -> None:
    initial = (ModelInput.of(MultimodalContent.of("initial")),)
    isolated = (ModelInput.of(MultimodalContent.of("isolated")),)
    calls = 0

    class ContextProvider:
        async def __call__(self) -> Sequence[ModelContextElement]:
            nonlocal calls
            calls += 1
            return isolated

    state = (
        await Step.appending_context(ModelOutput.of(MultimodalContent.of("temporary")))
        .with_isolated_context(ContextProvider())
        .process(initial)
    )

    assert calls == 1
    assert state.context == initial


async def _merge_first(branches: Iterable[StepState]) -> StepState:
    return next(iter(branches))


@pytest.mark.asyncio
async def test_concurrent_single_branch_failure_propagates_unwrapped() -> None:
    # branches run as tasks, so a failure arrives wrapped in the joining task
    # group's exception group - reporting a group of one would defeat every
    # caller catching the branch error, `with_retry` and `with_fallback` included
    @step
    async def failing(state: StepState) -> StepState:
        raise ValueError("branch boom")

    concurrent_step = Step.concurrent(
        Step.updating_artifacts(left=AlphaArtifact(value="L")),
        failing,
        merge=_merge_first,
    )

    with pytest.raises(ValueError, match="branch boom"):
        await concurrent_step.process()


@pytest.mark.asyncio
async def test_concurrent_branch_failure_is_catchable_by_wrappers() -> None:
    @step
    async def failing(state: StepState) -> StepState:
        raise ValueError("branch boom")

    @step
    async def fallback(state: StepState) -> StepState:
        return state.appending_context(ModelInput.of(MultimodalContent.of("fallback")))

    recovered = Step.concurrent(
        Step.updating_artifacts(left=AlphaArtifact(value="L")),
        failing,
        merge=_merge_first,
    ).with_fallback(fallback, catching=ValueError)

    state = await recovered.process()

    assert _text_of(state.context[-1].content) == "fallback"


@pytest.mark.asyncio
async def test_concurrent_multiple_branch_failures_raise_the_first_one() -> None:
    # the first branch failure ends the remaining branches and is raised on its own
    barrier = asyncio.Barrier(2)

    @step
    async def failing_value(state: StepState) -> StepState:
        await barrier.wait()
        raise ValueError("a")

    @step
    async def failing_key(state: StepState) -> StepState:
        await barrier.wait()
        raise KeyError("b")

    with pytest.raises((ValueError, KeyError)) as caught:
        await Step.concurrent(failing_value, failing_key, merge=_merge_first).process()

    assert not isinstance(caught.value, BaseExceptionGroup)


@pytest.mark.asyncio
async def test_concurrent_cancellation_stays_a_cancellation() -> None:
    # the task group cancels the branches on teardown - collecting those into a
    # group would turn a cancellation passing through into an ordinary failure
    @step
    async def slow(state: StepState) -> StepState:
        await asyncio.sleep(5)
        return state

    task = asyncio.ensure_future(
        Step.concurrent(slow, slow, merge=_merge_first).process(),
    )
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_concurrent_group_raised_by_merge_keeps_its_own_identity() -> None:
    # only the task group's own wrapper is unwrapped - a group raised by `merge`
    # is an error in its own right and keeps the message it was raised with
    async def group_merge(branches: Iterable[StepState]) -> StepState:
        raise ExceptionGroup("merge group", [ValueError("m1"), KeyError("m2")])

    with pytest.raises(BaseExceptionGroup) as caught:
        await Step.concurrent(
            Step.updating_artifacts(left=AlphaArtifact(value="L")),
            Step.updating_artifacts(right=BetaArtifact(value="R")),
            merge=group_merge,
        ).process()

    assert "merge group" in str(caught.value)
    assert {type(error) for error in caught.value.exceptions} == {ValueError, KeyError}


@pytest.mark.asyncio
async def test_concurrent_group_raised_by_branch_keeps_its_own_identity() -> None:
    @step
    async def failing(state: StepState) -> StepState:
        raise ExceptionGroup("branch group", [ValueError("b1"), KeyError("b2")])

    with pytest.raises(BaseExceptionGroup) as caught:
        await Step.concurrent(
            Step.updating_artifacts(left=AlphaArtifact(value="L")),
            failing,
            merge=_merge_first,
        ).process()

    assert "branch group" in str(caught.value)
    assert {type(error) for error in caught.value.exceptions} == {ValueError, KeyError}


@pytest.mark.asyncio
@pytest.mark.parametrize("looping", [False, True])
async def test_tool_output_context_groups_interleaved_requests(
    monkeypatch: pytest.MonkeyPatch,
    looping: bool,
) -> None:
    from collections.abc import AsyncGenerator, Collection

    from draive.tools import Toolbox

    requests = (
        ModelToolRequest.of("r1", tool="shared", arguments={}),
        ModelToolRequest.of("r2", tool="shared", arguments={}),
    )

    async def handle(
        self: Toolbox,
        pending: Collection[ModelToolRequest],
    ) -> AsyncGenerator[TextContent | ModelToolResponse]:
        assert tuple(pending) == requests
        yield TextContent.of("a1", meta={"request": "r2", "tool": "shared"})
        yield TextContent.of("b1", meta={"request": "r1", "tool": "shared"})
        yield TextContent.of("a2", meta={"request": "r2", "tool": "shared"})
        yield TextContent.of("b2", meta={"request": "r1", "tool": "shared"})
        for request in requests:
            yield ModelToolResponse(
                identifier=request.identifier,
                tool=request.tool,
                status="success",
                content=MultimodalContent.empty,
            )

    monkeypatch.setattr(Toolbox, "handle", handle)

    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        return _stream_of(*requests)

    if looping:
        state = (
            await Step.looping_completion()
            .with_ctx(GenerativeModel(generating=generating))
            .process()
        )
    else:
        state = await Step.handling_tools([]).process((ModelOutput.of(*requests),))

    assert len(state.context) == 3
    responses = state.context[-2]
    assert isinstance(responses, ModelInput)
    assert len(responses.tool_responses) == 2
    output = state.context[-1]
    assert isinstance(output, ModelOutput)
    assert _text_of(output.content) == "a1a2b1b2"
