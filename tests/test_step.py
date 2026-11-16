import asyncio
from collections.abc import AsyncIterable, Iterable, Sequence
from typing import Any

import pytest
from haiway import State, ctx

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
from draive.multimodal import ArtifactContent, MultimodalContent, TextContent
from draive.steps import (
    Step,
    StepException,
    StepRetriesExhausted,
    StepState,
    StepTimeout,
    step,
)
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
    from collections.abc import AsyncGenerator

    requests = (
        ModelToolRequest.of("r1", tool="shared", arguments={}),
        ModelToolRequest.of("r2", tool="shared", arguments={}),
    )

    async def merging(
        stream: AsyncGenerator[Any],
        /,
    ) -> AsyncGenerator[TextContent | ModelToolResponse]:
        await stream.aclose()  # the real execution is replaced by the interleaved output
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

    monkeypatch.setattr("draive.tools.toolbox._merging_output", merging)

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


@pytest.mark.asyncio
async def test_retry_restarts_from_original_state() -> None:
    seen_context_sizes: list[int] = []

    async def flaky(state: StepState) -> Any:
        seen_context_sizes.append(len(state.context))
        yield state.appending_context(ModelInput.of(MultimodalContent.of("partial")))
        if len(seen_context_sizes) == 1:
            raise ValueError("boom")

        yield state.appending_context(ModelInput.of(MultimodalContent.of("done")))

    result = await Step(flaky).with_retry(limit=1).process()

    assert seen_context_sizes == [0, 0]
    assert len(result.context) == 1
    assert _text_of(result.context[0].content) == "done"


@pytest.mark.asyncio
async def test_retry_without_state_output_discards_failed_attempt_state() -> None:
    attempts = 0

    async def flaky(state: StepState) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield state.appending_context(ModelInput.of(MultimodalContent.of("partial")))
            raise ValueError("boom")

    result = await Step(flaky).with_retry(limit=1).process()

    assert attempts == 2
    assert result.context == ()


@pytest.mark.asyncio
async def test_fallback_runs_from_original_state() -> None:
    seen: list[int] = []

    async def failing(state: StepState) -> Any:
        yield state.appending_context(ModelInput.of(MultimodalContent.of("partial")))
        raise ValueError("boom")

    async def fallback(state: StepState) -> Any:
        seen.append(len(state.context))
        yield state.appending_context(ModelInput.of(MultimodalContent.of("fallback")))

    result = await Step(failing).with_fallback(Step(fallback)).process()

    assert seen == [0]
    assert len(result.context) == 1
    assert _text_of(result.context[0].content) == "fallback"


@pytest.mark.asyncio
async def test_fallback_without_state_output_discards_failed_step_state() -> None:
    async def failing(state: StepState) -> Any:
        yield state.appending_context(ModelInput.of(MultimodalContent.of("partial")))
        raise ValueError("boom")

    initial = StepState.of((ModelInput.of(MultimodalContent.of("start")),))
    result = await Step(failing).with_fallback(Step.noop).process(initial)

    assert result == initial


@pytest.mark.asyncio
async def test_concurrent_merge_receives_reiterable_branches() -> None:
    counts: list[tuple[int, int]] = []

    async def merge(branches: Iterable[StepState]) -> StepState:
        counts.append((len(list(branches)), len(list(branches))))
        return StepState.of()

    await Step.concurrent(Step.noop, Step.noop, merge=merge).process()

    assert counts == [(2, 2)]


@pytest.mark.asyncio
async def test_volatile_tools_drops_elements_left_empty() -> None:
    tool_only = ModelOutput.of(ModelToolRequest.of("call", tool="lookup", arguments={}))
    mixed = ModelOutput.of(
        MultimodalContent.of("text"),
        ModelToolRequest.of("call2", tool="lookup", arguments={}),
    )

    state = await Step.appending_context(tool_only, mixed).with_volatile_tools().process()

    assert len(state.context) == 1
    assert _text_of(state.context[0].content) == "text"


@pytest.mark.asyncio
async def test_loop_with_false_initial_condition_keeps_state() -> None:
    async def never(state: StepState, iteration: int) -> bool:
        return False

    initial = (ModelInput.of(MultimodalContent.of("start")),)
    state = await Step.loop(Step.noop, condition=never).process(initial)

    assert state.context == initial


@pytest.mark.asyncio
async def test_process_rejects_artifacts_with_explicit_state() -> None:
    with pytest.raises(TypeError):
        await Step.noop.process(StepState.of(), AlphaArtifact(value="x"))

    with pytest.raises(TypeError):
        await Step.noop.run(StepState.of(), AlphaArtifact(value="x"))

    with pytest.raises(TypeError):
        stream = Step.noop.stream(StepState.of(), alpha=AlphaArtifact(value="x"))
        _ = [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_state_get_uses_explicit_key_even_when_empty() -> None:
    state = StepState.of((), **{"": AlphaArtifact(value="empty-key")})

    assert state.get(AlphaArtifact, key="") == AlphaArtifact(value="empty-key")
    assert state.get(AlphaArtifact) is None


class CtxMarker(State):
    value: str = "default"


class _MarkerDisposable:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def __aenter__(self) -> CtxMarker:
        self._events.append("enter")
        return CtxMarker(value="disposed")

    async def __aexit__(self, *_: Any) -> None:
        self._events.append("exit")


@pytest.mark.asyncio
@pytest.mark.parametrize("with_state", [False, True])
async def test_with_ctx_disposables_are_entered_and_released(with_state: bool) -> None:
    events: list[str] = []
    observed: list[str] = []

    @step
    async def observing(state: StepState) -> StepState:
        observed.append(ctx.state(CtxMarker).value)
        events.append("run")
        return state

    wrapped = observing.with_ctx(
        *((CtxMarker(value="explicit"),) if with_state else ()),
        disposables=(_MarkerDisposable(events),),
    )
    await wrapped.process()

    assert events == ["enter", "run", "exit"]
    assert observed == ["explicit" if with_state else "disposed"]


@pytest.mark.asyncio
async def test_output_evaluation_runs_once_with_complete_output() -> None:
    calls: list[int] = []

    async def evaluator(output: Sequence[Any]) -> EvaluatorResult:
        calls.append(len(output))
        return EvaluatorResult.of("output", score=1.0, threshold=0.5)

    await (
        Step.emitting(TextContent.of("a"), TextContent.of("b"), TextContent.of("c"))
        .with_output_evaluation(evaluator)
        .run()
    )

    assert calls == [3]


@pytest.mark.asyncio
async def test_output_evaluation_failure_carries_final_state() -> None:
    async def failing(_: Sequence[Any]) -> EvaluatorResult:
        return EvaluatorResult.of("output", score=0.0, threshold=1.0)

    delivered: list[Any] = []
    pipeline = Step.sequence(
        Step.emitting(TextContent.of("x")),
        Step.updating_artifacts(AlphaArtifact(value="final")),
    ).with_output_evaluation(failing, raise_on_failure=True)

    with pytest.raises(StepException) as exc_info:
        async for chunk in pipeline.stream():
            delivered.append(chunk)

    assert len(delivered) == 1  # output delivered before the evaluation failed
    assert exc_info.value.state.get(AlphaArtifact, required=True).value == "final"


@pytest.mark.asyncio
async def test_output_evaluation_not_run_when_step_fails() -> None:
    calls = 0

    async def evaluator(_: Sequence[Any]) -> EvaluatorResult:
        nonlocal calls
        calls += 1
        return EvaluatorResult.of("output", score=1.0, threshold=0.5)

    @step
    async def failing(state: StepState) -> StepState:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await failing.with_output_evaluation(evaluator).process()

    assert calls == 0


@pytest.mark.asyncio
async def test_duplicate_artifact_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="AlphaArtifact"):
        StepState.of((), AlphaArtifact(value="1"), AlphaArtifact(value="2"))

    with pytest.raises(ValueError, match="AlphaArtifact"):
        StepState.of((), AlphaArtifact(value="1"), AlphaArtifact=AlphaArtifact(value="2"))

    with pytest.raises(ValueError, match="AlphaArtifact"):
        StepState.of().updating_artifacts(AlphaArtifact(value="1"), AlphaArtifact(value="2"))

    with pytest.raises(ValueError, match="AlphaArtifact"):
        Step.updating_artifacts(AlphaArtifact(value="1"), AlphaArtifact(value="2"))


@pytest.mark.asyncio
async def test_updating_artifacts_replaces_existing_artifacts() -> None:
    state = StepState.of((), AlphaArtifact(value="old")).updating_artifacts(
        AlphaArtifact(value="new"),
        other=BetaArtifact(value="extra"),
    )

    assert state.get(AlphaArtifact, required=True).value == "new"
    assert state.get(BetaArtifact, key="other", required=True).value == "extra"


class ReportArtifact(State):
    title: str
    score: int


@pytest.mark.asyncio
async def test_loop_stops_with_exception_at_max_iterations() -> None:
    async def always(state: StepState, iteration: int) -> bool:
        return True

    executed = 0

    @step
    async def counting(state: StepState) -> StepState:
        nonlocal executed
        executed += 1
        return state

    with pytest.raises(StepException) as exc_info:
        await Step.loop(counting, condition=always, max_iterations=3).process()

    assert executed == 3
    assert exc_info.value.meta.get("max_iterations") == 3


@pytest.mark.asyncio
async def test_loop_ending_exactly_at_max_iterations_does_not_raise() -> None:
    async def twice(state: StepState, iteration: int) -> bool:
        return iteration < 2

    await Step.loop(Step.noop, condition=twice, max_iterations=2).process()


@pytest.mark.asyncio
async def test_loop_default_limit_is_32_and_can_be_disabled() -> None:
    async def always(state: StepState, iteration: int) -> bool:
        return True

    executed = 0

    @step
    async def counting(state: StepState) -> StepState:
        nonlocal executed
        executed += 1
        return state

    with pytest.raises(StepException):
        await Step.loop(counting, condition=always).process()

    assert executed == 32

    async def many(state: StepState, iteration: int) -> bool:
        return iteration < 40

    await Step.loop(Step.noop, condition=many, max_iterations=None).process()


@pytest.mark.parametrize("limit", [0, -1])
def test_max_iterations_must_be_positive(limit: int) -> None:
    async def never(state: StepState, iteration: int) -> bool:
        return False

    with pytest.raises(ValueError, match="max_iterations"):
        Step.loop(Step.noop, condition=never, max_iterations=limit)

    with pytest.raises(ValueError, match="max_iterations"):
        Step.looping_completion(max_iterations=limit)


@pytest.mark.asyncio
async def test_looping_completion_stops_at_max_iterations() -> None:
    @tool(description="Lookup")
    async def lookup() -> str:
        return "result"

    calls = 0

    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal calls
        calls += 1
        return _stream_of(ModelToolRequest.of(f"call-{calls}", tool="lookup", arguments={}))

    with pytest.raises(StepException) as exc_info:
        await (
            Step.looping_completion(tools=[lookup], max_iterations=2)
            .with_ctx(GenerativeModel(generating=generating))
            .process((ModelInput.of(MultimodalContent.of("go")),))
        )

    assert calls == 2
    assert isinstance(exc_info.value.state.context[-1], ModelInput)  # tool responses appended


@pytest.mark.asyncio
async def test_with_timeout_raises_step_timeout() -> None:
    async def slow(state: StepState) -> Any:
        await asyncio.sleep(1)
        yield state

    with pytest.raises(StepTimeout) as exc_info:
        await Step(slow).with_timeout(0.05).process()

    assert exc_info.value.timeout == 0.05
    assert isinstance(exc_info.value, StepException)


@pytest.mark.asyncio
async def test_with_timeout_excludes_consumer_time() -> None:
    async def emitting(state: StepState) -> Any:
        yield TextContent.of("a")
        yield TextContent.of("b")
        yield state

    received: list[str] = []
    async for chunk in Step(emitting).with_timeout(0.2).stream():
        await asyncio.sleep(0.3)  # slower than the limit, but outside of the step
        if isinstance(chunk, TextContent):
            received.append(chunk.text)

    assert received == ["a", "b"]


@pytest.mark.asyncio
async def test_with_timeout_accumulates_across_chunks() -> None:
    async def trickling(state: StepState) -> Any:
        for _ in range(10):
            await asyncio.sleep(0.05)
            yield TextContent.of("tick")

    with pytest.raises(StepTimeout):
        await Step(trickling).with_timeout(0.15).run()


@pytest.mark.asyncio
async def test_with_timeout_passes_through_fast_step_and_validates() -> None:
    result = await Step.emitting(TextContent.of("ok")).with_timeout(1).run()
    assert _text_of(result) == "ok"

    with pytest.raises(ValueError, match="positive"):
        Step.noop.with_timeout(0)


@pytest.mark.asyncio
async def test_timeout_is_retryable() -> None:
    attempts = 0

    async def sometimes_slow(state: StepState) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(1)

        yield TextContent.of("done")

    result = await Step(sometimes_slow).with_timeout(0.05).with_retry(limit=1).run()

    assert attempts == 2
    assert _text_of(result) == "done"


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_dedicated_exception() -> None:
    attempts = 0

    @step
    async def failing(state: StepState) -> StepState:
        nonlocal attempts
        attempts += 1
        raise ValueError(f"failure {attempts}")

    initial = StepState.of((ModelInput.of(MultimodalContent.of("start")),))
    with pytest.raises(StepRetriesExhausted) as exc_info:
        await failing.with_retry(limit=2).process(initial)

    assert attempts == 3
    assert exc_info.value.attempts == 3
    assert exc_info.value.state == initial
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert str(exc_info.value.__cause__) == "failure 3"


@pytest.mark.asyncio
async def test_retry_without_retries_or_match_raises_original_error() -> None:
    @step
    async def failing(state: StepState) -> StepState:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await failing.with_retry(limit=0).process()

    with pytest.raises(ValueError, match="boom"):
        await failing.with_retry(limit=3, catching=KeyError).process()


@pytest.mark.asyncio
async def test_retry_delay_variants() -> None:
    delays: list[tuple[int, str]] = []

    def delaying(attempt: int, exc: Exception) -> float:
        delays.append((attempt, str(exc)))
        return 0.0

    @step
    async def failing(state: StepState) -> StepState:
        raise ValueError("x")

    with pytest.raises(StepRetriesExhausted):
        await failing.with_retry(limit=2, delay=delaying).process()

    assert delays == [(1, "x"), (2, "x")]

    with pytest.raises(StepRetriesExhausted):
        await failing.with_retry(limit=1, delay=0.01).process()


@pytest.mark.asyncio
async def test_removing_artifacts() -> None:
    state = StepState.of(
        (),
        AlphaArtifact(value="a"),
        BetaArtifact(value="b"),
        custom=AlphaArtifact(value="c"),
    )

    by_type = state.removing_artifacts(AlphaArtifact)
    assert by_type.get(AlphaArtifact) is None
    assert by_type.get(BetaArtifact, required=True).value == "b"
    assert by_type.get(AlphaArtifact, key="custom", required=True).value == "c"

    by_key = state.removing_artifacts("custom", BetaArtifact, "missing")
    assert set(by_key.artifacts) == {"AlphaArtifact"}

    assert state.removing_artifacts("missing") is state
    assert state.removing_artifacts() is state

    stepped = await Step.removing_artifacts(AlphaArtifact).process(state)
    assert set(stepped.artifacts) == {"BetaArtifact", "custom"}
    assert Step.removing_artifacts() is Step.noop


@pytest.mark.asyncio
async def test_then_and_rshift_compose_sequences() -> None:
    first = Step.appending_context(ModelInput.of(MultimodalContent.of("1")))
    second = Step.appending_context(ModelOutput.of(MultimodalContent.of("2")))
    third = Step.emitting(TextContent.of("3"))

    via_then = first.then(second, third)
    via_operator = first >> second >> third

    for composed in (via_then, via_operator):
        state = await composed.process()
        assert [_text_of(element.content) for element in state.context] == ["1", "2"]
        assert _text_of(await composed.run()) == "3"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: ModelOutput.of(
                MultimodalContent.of('{"title": "Q3", "score": 7}'),
            ),
            id="json-text",
        ),
        pytest.param(
            lambda: ModelOutput.of(
                MultimodalContent.of('```json\n{"title": "Q3", "score": 7}\n```'),
            ),
            id="code-fence",
        ),
        pytest.param(
            lambda: ModelOutput.of(
                MultimodalContent.of(ArtifactContent.of(ReportArtifact(title="Q3", score=7))),
            ),
            id="artifact",
        ),
    ],
)
async def test_decoding_artifact_from_model_output(build: Any) -> None:
    state = await Step.decoding_artifact(ReportArtifact).process((build(),))

    assert state.get(ReportArtifact, required=True) == ReportArtifact(title="Q3", score=7)


@pytest.mark.asyncio
async def test_decoding_artifact_uses_explicit_key() -> None:
    output = ModelOutput.of(MultimodalContent.of('{"title": "Q3", "score": 7}'))

    state = await Step.decoding_artifact(ReportArtifact, key="report").process((output,))

    assert state.get(ReportArtifact, key="report", required=True).score == 7
    assert state.get(ReportArtifact) is None


@pytest.mark.asyncio
async def test_decoding_artifact_failures_raise_step_exception() -> None:
    with pytest.raises(StepException, match="ending with model output"):
        await Step.decoding_artifact(ReportArtifact).process()

    with pytest.raises(StepException, match="ending with model output"):
        await Step.decoding_artifact(ReportArtifact).process(
            (ModelInput.of(MultimodalContent.of("{}")),)
        )

    with pytest.raises(StepException, match="Failed to decode") as exc_info:
        await Step.decoding_artifact(ReportArtifact).process(
            (ModelOutput.of(MultimodalContent.of("not json")),)
        )

    assert exc_info.value.__cause__ is not None


@pytest.mark.asyncio
async def test_completion_with_typed_output_decodes_into_artifact() -> None:
    def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: Sequence[ModelContextElement],
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        assert output is ReportArtifact
        return _stream_of(TextContent.of('{"title": "Q3", "score": 7}'))

    pipeline = Step.looping_completion(output=ReportArtifact) >> Step.decoding_artifact(
        ReportArtifact
    )
    state = await pipeline.with_ctx(GenerativeModel(generating=generating)).process(
        (ModelInput.of(MultimodalContent.of("report")),)
    )

    assert state.get(ReportArtifact, required=True).title == "Q3"
