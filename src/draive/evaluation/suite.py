from asyncio import Lock, gather
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, Self, overload, runtime_checkable
from uuid import UUID, uuid4

from haiway import asynchronous, ctx, frozenlist

from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.evaluation.generator import generate_case_parameters
from draive.evaluation.scenario import PreparedScenarioEvaluator, ScenarioEvaluatorResult
from draive.parameters import DataModel, Field

__all__ = [
    "evaluation_suite",
    "EvaluationCaseResult",
    "EvaluationSuite",
    "EvaluationSuiteDefinition",
    "EvaluationSuiteStorage",
    "SuiteEvaluatorCaseResult",
    "SuiteEvaluatorResult",
]


class EvaluationSuiteCase[CaseParameters: DataModel](DataModel):
    identifier: UUID = Field(default_factory=uuid4)
    parameters: CaseParameters
    comment: str | None = None


class SuiteEvaluatorCaseResult[CaseParameters: DataModel, Value: DataModel | str](DataModel):
    case: EvaluationSuiteCase[CaseParameters] = Field(
        description="Evaluated case",
    )
    value: Value = Field(
        description="Evaluated value",
    )
    results: frozenlist[ScenarioEvaluatorResult] = Field(
        description="Evaluation results",
    )
    meta: Mapping[str, str | float | int | bool | None] | None = Field(
        description="Additional evaluation metadata",
        default=None,
    )

    @property
    def passed(self) -> bool:
        # empty results is equivalent of failure
        return len(self.results) > 0 and all(result.passed for result in self.results)

    def report(self) -> str:
        report: str = "\n---\n".join(
            [result.report() for result in self.results if not result.passed]
        )

        if report:  # nonempty report contains failing reports
            return (
                f"Evaluation case {self.case.identifier}:"
                f"\nvalue: {self.value}"
                f"\n{self.case.parameters}\n---\n{report}"
            )

        elif not self.results:
            return f"Evaluation case {self.case.identifier} empty!"

        else:
            return f"Evaluation case {self.case.identifier} passed!"


class SuiteEvaluatorResult[CaseParameters: DataModel, Value: DataModel | str](DataModel):
    cases: list[SuiteEvaluatorCaseResult[CaseParameters, Value]]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    def report(self) -> str:
        report: str = "\n---\n".join(
            [result.report() for result in self.cases if not result.passed]
        )

        if report:  # nonempty report contains failing reports
            return f"Evaluation suite failed:\n\n{report}"

        elif not self.cases:
            return "Evaluation suite empty!"

        else:
            return "Evaluation suite passed!"


class EvaluationCaseResult[Value: DataModel | str](DataModel):
    @classmethod
    def of(
        cls,
        results: ScenarioEvaluatorResult | EvaluatorResult,
        *_results: ScenarioEvaluatorResult | EvaluatorResult,
        value: Value,
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        free_results: list[EvaluatorResult] = []
        scenario_results: list[ScenarioEvaluatorResult] = []
        for result in (results, *_results):
            match result:
                case ScenarioEvaluatorResult() as scenario_result:
                    scenario_results.append(scenario_result)

                case EvaluatorResult() as evaluator_result:
                    free_results.append(evaluator_result)

        if free_results:
            scenario_results.append(
                ScenarioEvaluatorResult(
                    name="EvaluationSuite",
                    evaluations=tuple(free_results),
                )
            )

        return cls(
            value=value,
            results=tuple(scenario_results),
            meta=meta,
        )

    @classmethod
    async def evaluating(
        cls,
        value: Value,
        /,
        evaluators: PreparedScenarioEvaluator[Value] | PreparedEvaluator[Value],
        *_evaluators: PreparedScenarioEvaluator[Value] | PreparedEvaluator[Value],
        meta: Mapping[str, str | float | int | bool | None] | None = None,
    ) -> Self:
        return cls.of(
            *await gather(
                *[evaluator(value) for evaluator in [evaluators, *_evaluators]],
                return_exceptions=False,
            ),
            value=value,
            meta=meta,
        )

    value: Value = Field(
        description="Evaluated value",
    )
    results: frozenlist[ScenarioEvaluatorResult] = Field(
        description="Evaluation results",
    )
    meta: Mapping[str, str | float | int | bool | None] | None = Field(
        description="Additional evaluation metadata",
        default=None,
    )


@runtime_checkable
class EvaluationSuiteDefinition[CaseParameters: DataModel, Value: DataModel | str](Protocol):
    async def __call__(
        self,
        parameters: CaseParameters,
    ) -> EvaluationCaseResult[Value]: ...


class EvaluationSuiteData[CaseParameters: DataModel](DataModel):
    cases: frozenlist[EvaluationSuiteCase[CaseParameters]] = Field(default_factory=tuple)


@runtime_checkable
class EvaluationSuiteStorage[CaseParameters: DataModel](Protocol):
    async def load(
        self,
    ) -> EvaluationSuiteData[CaseParameters]: ...

    async def save(
        self,
        data: EvaluationSuiteData[CaseParameters],
    ) -> None: ...


class EvaluationSuite[CaseParameters: DataModel, Value: DataModel | str]:
    def __init__(
        self,
        definition: EvaluationSuiteDefinition[CaseParameters, Value],
        storage: EvaluationSuiteStorage[CaseParameters],
        parameters: type[CaseParameters],
    ) -> None:
        self._definition: EvaluationSuiteDefinition[CaseParameters, Value] = definition
        self._storage: EvaluationSuiteStorage[CaseParameters] = storage
        self._data_cache: EvaluationSuiteData[CaseParameters] | None = None
        self._parameters: type[CaseParameters] = parameters
        self._lock: Lock = Lock()

    @overload
    async def __call__(
        self,
        parameters: CaseParameters | UUID | None,
        /,
        *,
        reload: bool = False,
    ) -> SuiteEvaluatorCaseResult[CaseParameters, Value]: ...

    @overload
    async def __call__(
        self,
        /,
        *,
        reload: bool = False,
    ) -> SuiteEvaluatorResult[CaseParameters, Value]: ...

    async def __call__(
        self,
        parameters: CaseParameters | UUID | None = None,
        /,
        *,
        reload: bool = False,
    ) -> (
        SuiteEvaluatorResult[CaseParameters, Value]
        | SuiteEvaluatorCaseResult[CaseParameters, Value]
    ):
        async with self._lock:
            match parameters:
                case None:
                    return SuiteEvaluatorResult(
                        cases=await gather(
                            *[
                                self._evaluate(case=case)
                                for case in (await self._data(reload=reload)).cases
                            ],
                            return_exceptions=False,
                        )
                    )

                case UUID() as identifier:
                    available_cases: frozenlist[EvaluationSuiteCase[CaseParameters]] = (
                        await self._data(reload=reload)
                    ).cases

                    if evaluation_case := next(
                        iter([case for case in available_cases if case.identifier == identifier]),
                        None,
                    ):
                        return await self._evaluate(case=evaluation_case)

                    else:
                        raise ValueError(f"Evaluation case with ID {identifier} does not exists.")

                case case_parameters:
                    return await self._evaluate(
                        case=EvaluationSuiteCase[CaseParameters](
                            parameters=case_parameters,
                        )
                    )

    async def _evaluate(
        self,
        *,
        case: EvaluationSuiteCase[CaseParameters],
    ) -> SuiteEvaluatorCaseResult[CaseParameters, Value]:
        result: EvaluationCaseResult[Value] = await self._definition(parameters=case.parameters)

        return SuiteEvaluatorCaseResult[CaseParameters, Value](
            case=case,
            value=result.value,
            results=result.results,
        )

    async def _data(
        self,
        reload: bool = False,
    ) -> EvaluationSuiteData[CaseParameters]:
        if reload or self._data_cache is None:
            self._data_cache = await self._storage.load()
            return self._data_cache

        else:
            return self._data_cache

    async def cases(
        self,
        reload: bool = False,
    ) -> frozenlist[EvaluationSuiteCase[CaseParameters]]:
        async with self._lock:
            return (await self._data(reload=reload)).cases

    async def add_case(
        self,
        parameters: CaseParameters,
        /,
        comment: str | None = None,
    ) -> None:
        async with self._lock:
            data: EvaluationSuiteData[CaseParameters] = await self._data(reload=True)
            self._data_cache = data.updated(
                cases=(
                    *data.cases,
                    EvaluationSuiteCase[CaseParameters](
                        parameters=parameters,
                        comment=comment,
                    ),
                )
            )
            await self._storage.save(self._data_cache)

    async def generate_cases(
        self,
        *,
        persist: bool = False,
        count: int,
        examples: Iterable[CaseParameters] | None = None,
    ) -> None:
        async with self._lock:
            data: EvaluationSuiteData[CaseParameters] = await self._data(reload=True)
            self._data_cache = data.updated(
                cases=(
                    *data.cases,
                    *[
                        EvaluationSuiteCase(parameters=parameters)
                        for parameters in await generate_case_parameters(
                            self._parameters,
                            count=count,
                            examples=examples or [case.parameters for case in data.cases],
                        )
                    ],
                )
            )

            if persist:
                await self._storage.save(self._data_cache)

    async def remove_case(
        self,
        identifier: UUID,
        /,
    ) -> None:
        async with self._lock:
            data: EvaluationSuiteData[CaseParameters] = await self._data(reload=True)
            self._data_cache = data.updated(
                cases=tuple(case for case in data.cases if case.identifier != identifier)
            )
            await self._storage.save(self._data_cache)


def evaluation_suite[CaseParameters: DataModel, Value: DataModel | str](
    case: type[CaseParameters],
    /,
    storage: EvaluationSuiteStorage[CaseParameters] | Path | str | None = None,
) -> Callable[
    [EvaluationSuiteDefinition[CaseParameters, Value]],
    EvaluationSuite[CaseParameters, Value],
]:
    suite_storage: EvaluationSuiteStorage[CaseParameters]
    match storage:
        case None:
            suite_storage = _EvaluationSuiteMemoryStorage[CaseParameters](
                data_type=EvaluationSuiteData[case],
            )

        case str() as path_str:
            suite_storage = _EvaluationSuiteFileStorage[CaseParameters](
                path=path_str,
                data_type=EvaluationSuiteData[case],
            )

        case Path() as path:
            suite_storage = _EvaluationSuiteFileStorage[CaseParameters](
                path=path,
                data_type=EvaluationSuiteData[case],
            )

        case storage:
            suite_storage = storage

    def wrap(
        definition: EvaluationSuiteDefinition[CaseParameters, Value],
    ) -> EvaluationSuite[CaseParameters, Value]:
        return EvaluationSuite[CaseParameters, Value](
            definition=definition,
            storage=suite_storage,
            parameters=case,
        )

    return wrap


class _EvaluationSuiteMemoryStorage[CaseParameters: DataModel]:
    def __init__(
        self,
        data_type: type[EvaluationSuiteData[CaseParameters]],
    ) -> None:
        self.store: frozenlist[EvaluationSuiteCase[CaseParameters]] = ()
        self._data_type: type[EvaluationSuiteData[CaseParameters]] = data_type

    async def load(
        self,
    ) -> EvaluationSuiteData[CaseParameters]:
        return self._data_type(cases=self.store)

    async def save(
        self,
        data: EvaluationSuiteData[CaseParameters],
    ) -> None:
        self.store = data.cases


class _EvaluationSuiteFileStorage[CaseParameters: DataModel]:
    def __init__(
        self,
        path: Path | str,
        data_type: type[EvaluationSuiteData[CaseParameters]],
    ) -> None:
        self._path: Path
        match path:
            case str() as path_str:
                self._path = Path(path_str)

            case path:
                self._path = path

        self._data_type: type[EvaluationSuiteData[CaseParameters]] = data_type

    async def load(
        self,
    ) -> EvaluationSuiteData[CaseParameters]:
        try:
            return await self._file_load(data_type=self._data_type)

        except ValueError as exc:
            ctx.log_error(
                f"Invalid EvaluationSuite at {self._path}",
                exception=exc,
            )
            return self._data_type()

    @asynchronous
    def _file_load(
        self,
        data_type: type[EvaluationSuiteData[CaseParameters]],
    ) -> EvaluationSuiteData[CaseParameters]:
        if self._path.exists():
            with open(self._path, mode="rb") as file:
                return data_type.from_json(file.read())

        else:
            return data_type()

    async def save(
        self,
        data: EvaluationSuiteData[CaseParameters],
    ) -> None:
        await self._file_save(data=data)

    @asynchronous
    def _file_save(
        self,
        data: EvaluationSuiteData[CaseParameters],
    ) -> None:
        with open(self._path, mode="wb+") as file:
            file.write(data.as_json(indent=2).encode("utf-8"))
