from collections.abc import Sequence

import pytest
from haiway import State, ctx

from draive.evaluation import (
    EvaluatorResult,
    EvaluatorScenarioResult,
    EvaluatorSuiteResult,
    evaluator_scenario,
    evaluator_suite,
)
from draive.evaluation.suite import EvaluatorSuiteCase, EvaluatorSuiteCaseResult


class _Params(State):
    value: int


def _result(score: float, threshold: float = 0.5, name: str = "check") -> EvaluatorResult:
    return EvaluatorResult.of(name, score=score, threshold=threshold)


def _suite(*cases: EvaluatorSuiteCase[_Params]):
    async def definition(
        parameters: _Params,
    ) -> Sequence[EvaluatorScenarioResult | EvaluatorResult]:
        return [_result(parameters.value / 10)]

    return evaluator_suite(_Params, storage=cases, name="sample")(definition)


# --- scenario -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario_failure_is_kept_as_failed_result_with_error() -> None:
    @evaluator_scenario(name="exploding")
    async def exploding(value: str) -> Sequence[EvaluatorResult]:
        raise ValueError("kaboom")

    async with ctx.scope("test"):
        result = await exploding("x")

    assert not result.passed
    assert result.performance == 0.0
    assert "kaboom" in str(result.results[0].meta["error"])


@pytest.mark.asyncio
async def test_scenario_without_results_never_passes() -> None:
    @evaluator_scenario(name="empty")
    async def empty(value: str) -> Sequence[EvaluatorResult]:
        return []

    async with ctx.scope("test"):
        result = await empty("x")

    assert not result.passed
    assert result.performance == 0.0
    assert result.report() == "empty has no results"


@pytest.mark.asyncio
async def test_scenario_prepared_binds_arguments() -> None:
    @evaluator_scenario(name="bound")
    async def bound(value: str, *, score: float) -> Sequence[EvaluatorResult]:
        return [_result(score)]

    async with ctx.scope("test"):
        result = await bound.prepared(score=0.9)("x")

    assert result.passed
    assert result.results[0].score == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("results", "performance"),
    (
        ((_result(1.0, 0.5), _result(0.0, 0.5)), 50.0),  # overshoot is capped per result
        ((_result(0.25, 0.5),), 50.0),
        ((_result(1.0, 0.1),), 100.0),
    ),
)
def test_scenario_result_performance_caps_each_result(
    results: Sequence[EvaluatorResult],
    performance: float,
) -> None:
    assert EvaluatorScenarioResult(scenario="s", results=results).performance == pytest.approx(
        performance
    )


def test_scenario_name_with_spaces_is_normalized_like_evaluator_name() -> None:
    async def definition(value: str) -> Sequence[EvaluatorResult]:
        return []

    assert evaluator_scenario(name="My Scenario")(definition).name == "my_scenario"


def test_scenario_result_json_round_trip() -> None:
    result = EvaluatorScenarioResult(scenario="s", results=(_result(0.4), _result(0.9)))

    restored = EvaluatorScenarioResult.from_json(result.to_json())

    assert restored == result
    assert restored.passed is result.passed


# --- suite ----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", (0, -1, 0.0, -0.5, 1.5, float("nan")))
async def test_suite_rejects_out_of_range_selection(selection: float) -> None:
    suite = _suite(EvaluatorSuiteCase(identifier="a", parameters=_Params(value=1)))

    async with ctx.scope("test"):
        with pytest.raises(ValueError):
            await suite(selection)


@pytest.mark.asyncio
async def test_suite_unknown_identifier_raises() -> None:
    suite = _suite(EvaluatorSuiteCase(identifier="a", parameters=_Params(value=1)))

    async with ctx.scope("test"):
        with pytest.raises(ValueError, match="zzz"):
            await suite(["zzz"])


@pytest.mark.asyncio
async def test_suite_selection_by_identifier_parameters_and_fraction() -> None:
    suite = _suite(
        EvaluatorSuiteCase(identifier="a", parameters=_Params(value=10)),
        EvaluatorSuiteCase(identifier="b", parameters=_Params(value=1)),
    )

    async with ctx.scope("test"):
        by_id = await suite(["b"])
        by_params = await suite([_Params(value=7)])
        tiny_fraction = await suite(0.01)
        everything = await suite()
        count_over = await suite(99)

    assert [case.case_identifier for case in by_id.results] == ["b"]
    assert not by_id.passed
    assert by_params.passed
    assert len(tiny_fraction.results) == 1
    assert len(everything.results) == 2
    assert len(count_over.results) == 2


@pytest.mark.asyncio
async def test_empty_suite_selection_never_passes() -> None:
    suite = _suite(EvaluatorSuiteCase(identifier="a", parameters=_Params(value=10)))

    async with ctx.scope("test"):
        result = await suite([])

    assert result.results == ()
    assert not result.passed
    assert result.performance == 0.0
    assert result.report() == "Evaluator suite empty!"


def test_suite_result_round_trip_keeps_result_kinds() -> None:
    result = EvaluatorSuiteResult(
        suite="s",
        results=(
            EvaluatorSuiteCaseResult(
                case_identifier="a",
                results=(
                    _result(1.0),
                    EvaluatorScenarioResult(scenario="sc", results=(_result(0.2),)),
                ),
            ),
        ),
    )

    restored = EvaluatorSuiteResult.from_json(result.to_json())

    assert restored == result
    assert [type(item) for item in restored.results[0].results] == [
        EvaluatorResult,
        EvaluatorScenarioResult,
    ]


def test_case_result_brief_report_formats_performance() -> None:
    case = EvaluatorSuiteCaseResult(
        case_identifier="a",
        results=(_result(1.0), _result(1.0), _result(0.0)),
    )

    assert case.report(detailed=False).startswith("a: failed (66.67%)")


@pytest.mark.asyncio
async def test_file_storage_round_trips_added_cases(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text("[]")
    suite = _suite().with_storage(path)

    async with ctx.scope("test"):
        await suite.add_case(_Params(value=3), "x")
        reloaded = await suite.cases(reload=True)
        await suite.remove_case("x")
        after_remove = await suite.cases(reload=True)

    assert [(case.identifier, case.parameters.value) for case in reloaded] == [("x", 3)]
    assert after_remove == ()


@pytest.mark.asyncio
async def test_file_storage_add_case_creates_missing_file(tmp_path) -> None:
    path = tmp_path / "nested" / "new_cases.json"
    suite = _suite().with_storage(path)

    async with ctx.scope("test"):
        await suite.add_case(_Params(value=3), "x")

    assert path.exists()


@pytest.mark.asyncio
async def test_add_case_propagates_missing_data_of_custom_storage() -> None:
    saved: list[Sequence[EvaluatorSuiteCase[_Params]]] = []

    class _Storage:
        async def load(self) -> Sequence[EvaluatorSuiteCase[_Params]]:
            raise FileNotFoundError("remote object not found")

        async def save(self, cases: Sequence[EvaluatorSuiteCase[_Params]]) -> None:
            saved.append(cases)

    suite = _suite().with_storage(_Storage())

    async with ctx.scope("test"):
        with pytest.raises(FileNotFoundError):
            await suite.add_case(_Params(value=3), "x")

    assert saved == []


@pytest.mark.asyncio
async def test_add_case_rejects_duplicate_identifier() -> None:
    suite = _suite(EvaluatorSuiteCase(identifier="a", parameters=_Params(value=10)))

    async with ctx.scope("test"):
        with pytest.raises(ValueError):
            await suite.add_case(_Params(value=1), "a")


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ("{not json", "{}", '[{"identifier": "a", "parameters": {}}]'))
async def test_file_storage_invalid_content_raises(tmp_path, content: str) -> None:
    path = tmp_path / "cases.json"
    path.write_text(content)
    suite = _suite().with_storage(path)

    async with ctx.scope("test"):
        with pytest.raises(Exception):  # noqa: B017
            await suite()


@pytest.mark.asyncio
async def test_file_storage_bootstrap_leaves_valid_file_and_no_temporary_files(tmp_path) -> None:
    path = tmp_path / "cases.json"
    suite = _suite().with_storage(path)

    async with ctx.scope("test"):
        await suite.add_case(_Params(value=3), "x")
        reloaded = await suite.cases(reload=True)

    assert [case.identifier for case in reloaded] == ["x"]
    assert [item.name for item in tmp_path.iterdir()] == ["cases.json"]


@pytest.mark.asyncio
async def test_file_storage_missing_file_still_fails_without_add(tmp_path) -> None:
    suite = _suite().with_storage(tmp_path / "missing.json")

    async with ctx.scope("test"):
        with pytest.raises(FileNotFoundError):
            await suite()


@pytest.mark.asyncio
async def test_scenario_raises_for_invalid_arguments_instead_of_scoring() -> None:
    @evaluator_scenario(name="needs_reference")
    async def needs_reference(value: str, *, reference: str) -> Sequence[EvaluatorResult]:
        return [_result(1.0)]

    async with ctx.scope("test"):
        with pytest.raises(TypeError):
            await needs_reference("x")  # pyright: ignore[reportCallIssue]

        with pytest.raises(TypeError):
            await needs_reference("x", reference="r", extra=1)  # pyright: ignore[reportCallIssue]

        result = await needs_reference("x", reference="r")

    assert result.passed


@pytest.mark.asyncio
async def test_scenario_type_error_raised_inside_definition_is_a_failed_result() -> None:
    @evaluator_scenario(name="inner_type_error")
    async def inner_type_error(value: str) -> Sequence[EvaluatorResult]:
        raise TypeError("inner")

    async with ctx.scope("test"):
        result = await inner_type_error("x")

    assert not result.passed
    assert "inner" in str(result.results[0].meta["error"])


def test_reports_escape_xml_special_characters() -> None:
    hostile = EvaluatorResult.of(
        "it's <b>&",
        score=0.5,
        threshold=0.5,
        meta={"comment": "</meta><x a='1'>&"},
    )
    scenario = EvaluatorScenarioResult(scenario="s'<>", results=(hostile,))
    case = EvaluatorSuiteCaseResult(
        case_identifier="c'&",
        results=(scenario,),
    )

    for report in (hostile.report(), scenario.report(), case.report()):
        assert "<b>" not in report
        assert "<x a=" not in report
        assert "</meta><x" not in report
        assert "&amp;" in report


@pytest.mark.asyncio
async def test_suite_selection_int_is_count_float_is_fraction_and_bool_is_rejected() -> None:
    suite = _suite(
        *(
            EvaluatorSuiteCase(identifier=str(index), parameters=_Params(value=index))
            for index in range(4)
        )
    )

    async with ctx.scope("test"):
        one_case = await suite(1)
        all_cases = await suite(1.0)
        half = await suite(0.5)
        with pytest.raises(ValueError):
            await suite(True)

    assert len(one_case.results) == 1
    assert len(all_cases.results) == 4
    assert len(half.results) == 2
