import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import pytest
from haiway import State, ctx
from haiway.context.types import ContextStateMissing

from draive import (
    DecisionAnswer,
    DecisionChoice,
    DecisionChoiceAnswer,
    DecisionGeneration,
    DecisionModel,
    DecisionQuestion,
    GenerativeModel,
    ModelInput,
    ModelOutput,
    ModelReasoning,
    ModelToolRequest,
    ModelToolResponse,
    Multimodal,
    Step,
    StepException,
    StepState,
    generate_completion_decisions,
    step,
)
from draive.models import ModelContext, ModelOutputChunk, ModelTools
from draive.multimodal import MultimodalContent, TextContent


class Marker(State):
    value: str


class Seen(State):
    value: str


class Calls:
    def __init__(self) -> None:
        self.inputs: list[Any] = []
        self.questions: list[DecisionChoice] = []
        self.extras: list[dict[str, Any]] = []


def _decider(
    *chosen: str,
    calls: Calls | None = None,
) -> DecisionModel:
    remaining: list[str] = list(chosen)

    async def generating(
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        question = questions[0]
        assert isinstance(question, DecisionChoice)
        if calls is not None:
            calls.inputs.append(input)
            calls.questions.append(question)
            calls.extras.append(extra)

        identifier: str = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return (
            DecisionChoiceAnswer(
                identifier=question.identifier,
                choice=next(
                    (option for option in question.options if option.identifier == identifier),
                    question.options[0].updating(identifier=identifier),
                ),
            ),
        )

    return DecisionModel(generating=generating)


def _marking(value: str, *, name: str | None = None, description: str | None = None) -> Step:
    marking = Step.updating_artifacts(Marker(value=value))
    if name is not None:
        marking = marking.with_meta(name=name)

    if description is not None:
        marking = marking.with_meta(description=description)

    return marking


def _routing(**arguments: Any) -> Step:
    return Step.deciding(
        "which one fits",
        _marking("A", name="alpha"),
        _marking("B", name="beta"),
        **arguments,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("chosen", ["alpha", "beta"])
async def test_executes_only_the_chosen_step(chosen: str) -> None:
    executed: list[str] = []

    async def alpha(state: StepState) -> StepState:
        executed.append("alpha")
        return state

    async def beta(state: StepState) -> StepState:
        executed.append("beta")
        return state

    routing = Step.deciding(
        "c",
        step(alpha).with_meta(name="alpha"),
        step(beta).with_meta(name="beta"),
    )
    async with ctx.scope("test", _decider(chosen)):
        await routing.process()

    assert executed == [chosen]


@pytest.mark.asyncio
async def test_decides_on_every_execution() -> None:
    calls = Calls()
    routing = _routing()

    async with ctx.scope("test", _decider("alpha", "beta", calls=calls)):
        first = await routing.process()
        second = await routing.process()

    assert first.get(Marker, required=True).value == "A"
    assert second.get(Marker, required=True).value == "B"
    assert len(calls.inputs) == 2


@pytest.mark.asyncio
async def test_question_carries_criteria_and_step_names_and_descriptions() -> None:
    calls = Calls()
    routing = Step.deciding(
        "which one fits",
        _marking("A", name="alpha"),
        _marking("B", name="beta", description="when beta fits"),
    )

    async with ctx.scope("test", _decider("alpha", calls=calls)):
        await routing.process()

    question = calls.questions[0]
    assert question.criteria == "which one fits"
    assert [(option.identifier, option.description) for option in question.options] == [
        ("alpha", None),
        ("beta", "when beta fits"),
    ]
    assert calls.extras == [{}]


@pytest.mark.asyncio
async def test_default_input_renders_the_whole_context_with_roles_in_order() -> None:
    calls = Calls()
    context = (
        ModelInput.of(MultimodalContent.of("question")),
        ModelOutput.of(MultimodalContent.of("answer")),
        ModelInput.of(MultimodalContent.of("follow up")),
    )

    async with ctx.scope("test", _decider("alpha", calls=calls)):
        await _routing().process(context)

    rendered = str(calls.inputs[0])
    assert rendered.index("question") < rendered.index("answer") < rendered.index("follow up")
    assert rendered.count("<user") == 2
    assert rendered.count("<assistant") == 1


@pytest.mark.asyncio
async def test_default_input_of_empty_context_is_empty() -> None:
    calls = Calls()

    async with ctx.scope("test", _decider("alpha", calls=calls)):
        await _routing().process()

    assert calls.inputs[0].to_str() == ""


@pytest.mark.asyncio
async def test_custom_input_receives_the_state() -> None:
    calls = Calls()
    received: list[StepState] = []

    async def preparing(state: StepState) -> str:
        received.append(state)
        return f"marker={state.get(Marker, required=True).value}"

    async with ctx.scope("test", _decider("alpha", calls=calls)):
        await _routing(input=preparing).process((), Marker(value="m"))

    assert calls.inputs == ["marker=m"]
    assert len(received) == 1


@pytest.mark.asyncio
async def test_chosen_step_runs_with_the_state_of_the_decision() -> None:
    async def seen(state: StepState) -> StepState:
        return state.updating_artifacts(Seen(value=state.get(Marker, required=True).value))

    routing = Step.deciding(
        "c",
        step(seen).with_meta(name="alpha"),
        _marking("B", name="beta"),
    )
    async with ctx.scope("test", _decider("alpha")):
        state = await routing.process((), Marker(value="carried"))

    assert state.get(Seen, required=True).value == "carried"


@pytest.mark.asyncio
async def test_output_of_the_chosen_step_is_forwarded() -> None:
    routing = Step.deciding(
        "c",
        Step.appending_output("from alpha", emitting=True).with_meta(name="alpha"),
        Step.appending_output("from beta", emitting=True).with_meta(name="beta"),
    )

    async with ctx.scope("test", _decider("beta")):
        result = await routing.run()

    assert result.to_str() == "from beta"


@pytest.mark.asyncio
async def test_composes_in_sequences_and_loops() -> None:
    calls = Calls()
    pipeline = Step.sequence(
        _marking("start"),
        _routing(),
        Step.updating_artifacts(Seen(value="after")),
    )

    async with ctx.scope("test", _decider("beta", calls=calls)):
        state = await pipeline.process()

    assert state.get(Marker, required=True).value == "B"
    assert state.get(Seen, required=True).value == "after"

    iterations: list[int] = []

    async def until(state: StepState, iteration: int) -> bool:
        iterations.append(iteration)
        return iteration < 2

    looping = Step.loop(_routing(), condition=until)
    calls = Calls()
    async with ctx.scope("test", _decider("alpha", calls=calls)):
        await looping.process()

    assert iterations == [0, 1, 2]
    assert len(calls.inputs) == 2  # decided on every executed iteration


@pytest.mark.asyncio
async def test_nested_decisions_decide_independently() -> None:
    inner = Step.deciding("inner", _marking("X", name="x"), _marking("Y", name="y"))
    outer = Step.deciding(
        "outer",
        inner.with_meta(name="deep"),
        _marking("flat", name="flat"),
    )
    calls = Calls()

    async with ctx.scope("test", _decider("deep", "y", calls=calls)):
        state = await outer.process()

    assert state.get(Marker, required=True).value == "Y"
    assert [question.criteria for question in calls.questions] == ["outer", "inner"]


@pytest.mark.asyncio
async def test_unknown_option_from_backend_fails_with_step_exception() -> None:
    async def generating(
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        question = questions[0]
        assert isinstance(question, DecisionChoice)
        return (
            DecisionChoiceAnswer(
                identifier=question.identifier,
                choice=question.options[0].updating(identifier="gamma"),
            ),
        )

    async with ctx.scope("test", DecisionModel(generating=generating)):
        with pytest.raises(StepException, match="gamma") as raised:
            await _routing().process((), Marker(value="kept"))

    assert raised.value.state.get(Marker, required=True).value == "kept"


@pytest.mark.asyncio
async def test_backend_failure_is_propagated() -> None:
    async def generating(
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        raise RuntimeError("backend down")

    async with ctx.scope("test", DecisionModel(generating=generating)):
        with pytest.raises(RuntimeError, match="backend down"):
            await _routing().process()


@pytest.mark.asyncio
async def test_failure_can_be_recovered_with_fallback() -> None:
    async def generating(
        *,
        input: Multimodal,  # noqa: A002
        questions: Sequence[DecisionQuestion],
        **extra: Any,
    ) -> Sequence[DecisionAnswer]:
        raise StepException("no decision", state=StepState.of())

    routing = _routing().with_fallback(_marking("fallback"))
    async with ctx.scope("test", DecisionModel(generating=generating)):
        state = await routing.process()

    assert state.get(Marker, required=True).value == "fallback"


@pytest.mark.asyncio
async def test_requires_a_bound_decision_model() -> None:
    async with ctx.scope("test", DecisionGeneration(generating=_never)):
        with pytest.raises(ContextStateMissing):
            await _routing().process()


async def _never(
    *,
    input: Multimodal,  # noqa: A002
    questions: Sequence[DecisionQuestion],
    **extra: Any,
) -> Sequence[DecisionAnswer]:
    raise AssertionError("must not be used")


def test_deciding_requires_at_least_two_steps() -> None:
    with pytest.raises(ValueError, match="at least two"):
        Step.deciding("c")

    with pytest.raises(ValueError, match="at least two"):
        Step.deciding("c", _marking("A", name="alpha"))


@pytest.mark.parametrize("name", [None, "", " padded"])
def test_deciding_requires_a_valid_name_of_every_step(name: str | None) -> None:
    unnamed = _marking("B") if name is None else _marking("B").with_meta(name=name)

    with pytest.raises(ValueError):
        Step.deciding("c", _marking("A", name="alpha"), unnamed)


def test_deciding_reports_the_index_of_the_unnamed_step() -> None:
    with pytest.raises(ValueError, match="index 1"):
        Step.deciding("c", _marking("A", name="alpha"), _marking("B"))


def test_deciding_requires_unique_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        Step.deciding("c", _marking("A", name="alpha"), _marking("B", name="alpha"))


def test_deciding_requires_criteria() -> None:
    with pytest.raises(ValueError, match="criteria"):
        Step.deciding("", _marking("A", name="alpha"), _marking("B", name="beta"))


@pytest.mark.asyncio
async def test_step_modifiers_keep_the_name_used_for_deciding() -> None:
    routing = Step.deciding(
        "c",
        _marking("A", name="alpha").with_retry(limit=2).with_timeout(5),
        _marking("B", name="beta").with_fallback(_marking("F")).with_volatile_tools(),
    )

    async with ctx.scope("test", _decider("beta")):
        state = await routing.process()

    assert state.get(Marker, required=True).value == "B"


@pytest.mark.asyncio
async def test_works_with_explicit_completion_based_decisions() -> None:
    contexts: list[ModelContext] = []

    async def generating(
        *,
        instructions: str,
        tools: ModelTools,
        context: ModelContext,
        output: Any,
        **extra: Any,
    ) -> AsyncGenerator[ModelOutputChunk]:
        contexts.append(context)
        yield TextContent.of(json.dumps({"reasoning": "because", "choice": "beta"}))

    model = GenerativeModel(generating=generating)
    async with ctx.scope(
        "test",
        model,
        DecisionModel(generating=generate_completion_decisions),
    ):
        state = await _routing().process((ModelInput.of(MultimodalContent.of("hello")),))

    assert state.get(Marker, required=True).value == "B"
    assert len(contexts) == 1


@pytest.mark.asyncio
async def test_completion_based_decisions_are_never_implicit() -> None:
    async def generating(**_: Any) -> AsyncGenerator[ModelOutputChunk]:
        raise AssertionError("completions must not be used implicitly")
        yield TextContent.of("")

    async with ctx.scope("test", GenerativeModel(generating=generating)):
        with pytest.raises(ContextStateMissing):
            await _routing().process()


def test_step_meta_is_empty_by_default_and_merged_by_with_meta() -> None:
    plain = _marking("A")
    named = plain.with_meta(name="alpha").with_meta({"description": "first", "extra": 1})

    assert not plain.meta
    assert named.meta.name == "alpha"
    assert named.meta.description == "first"
    assert named.meta.get("extra") == 1
    assert named.with_meta(name="renamed").meta.name == "renamed"
    assert named.meta.name == "alpha"  # steps are immutable


@pytest.mark.asyncio
async def test_with_meta_does_not_change_execution() -> None:
    state = await _marking("A").with_meta(name="alpha").process()

    assert state.get(Marker, required=True).value == "A"


@pytest.mark.parametrize(
    "modify",
    [
        lambda step: step.with_retry(limit=2),
        lambda step: step.with_timeout(5),
        lambda step: step.with_fallback(Step.updating_artifacts(Seen(value="f"))),
        lambda step: step.with_condition(True),
        lambda step: step.with_volatile_context(),
        lambda step: step.with_suppressed_output(),
        lambda step: step.with_ctx(),
    ],
)
def test_wrapping_modifiers_keep_the_meta(modify: Any) -> None:
    described = _marking("A", name="alpha", description="first")

    modified = modify(described)

    assert modified.meta.name == "alpha"
    assert modified.meta.description == "first"


def test_combining_steps_starts_with_empty_meta() -> None:
    described = _marking("A", name="alpha")

    assert not Step.sequence(described, described).meta
    assert not described.then(described).meta
    assert not Step.deciding("c", described, _marking("B", name="beta")).meta


def test_deciding_does_not_drop_a_step_sharing_a_name() -> None:
    with pytest.raises(ValueError, match="unique"):
        Step.deciding(
            "c",
            _marking("A", name="alpha"),
            _marking("B", name="beta"),
            _marking("C", name="alpha"),
        )


@pytest.mark.asyncio
async def test_default_input_carries_tool_requests_and_responses() -> None:
    suspended = Calls()
    active = Calls()

    def context(result: str) -> tuple[ModelInput | ModelOutput, ...]:
        return (
            ModelInput.of(MultimodalContent.of("check my account")),
            ModelOutput.of(
                ModelToolRequest.of("call-1", tool="account_status", arguments={"user": "u1"}),
            ),
            ModelInput.of(
                ModelToolResponse.of(
                    "call-1",
                    tool="account_status",
                    content=MultimodalContent.of(result),
                ),
            ),
        )

    async with ctx.scope("test", _decider("alpha", calls=suspended)):
        await _routing().process(context("account suspended"))

    async with ctx.scope("test", _decider("alpha", calls=active)):
        await _routing().process(context("account active"))

    first = suspended.inputs[0].to_str()
    assert "account suspended" in first
    assert "account active" not in first
    assert "account_status" in first  # both the request and the response name the tool
    assert '"user": "u1"' in first
    assert first.index("check my account") < first.index("account_status")
    assert first != active.inputs[0].to_str()


@pytest.mark.asyncio
async def test_default_input_skips_reasoning() -> None:
    calls = Calls()
    context = (
        ModelOutput.of(
            ModelReasoning.of(MultimodalContent.of("secret thoughts")),
            MultimodalContent.of("visible answer"),
        ),
    )

    async with ctx.scope("test", _decider("alpha", calls=calls)):
        await _routing().process(context)

    rendered = calls.inputs[0].to_str()
    assert "visible answer" in rendered
    assert "secret thoughts" not in rendered


class _FalsyPreparer:
    def __bool__(self) -> bool:
        return False

    async def __call__(self, state: StepState) -> str:
        return "prepared"


@pytest.mark.asyncio
async def test_explicit_input_is_used_regardless_of_its_truthiness() -> None:
    calls = Calls()

    async with ctx.scope("test", _decider("alpha", calls=calls)):
        await _routing(input=_FalsyPreparer()).process(
            (ModelInput.of(MultimodalContent.of("context")),),
        )

    assert calls.inputs == ["prepared"]
