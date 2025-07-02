from asyncio import Lock, gather
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol, Self, overload, runtime_checkable
from uuid import UUID, uuid4

from haiway import ScopeContext, asynchronous, ctx

from draive.commons import META_EMPTY, Meta, MetaValues
from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.evaluation.generator import generate_case_parameters
from draive.evaluation.scenario import PreparedScenarioEvaluator, ScenarioEvaluatorResult
from draive.parameters import DataModel, Field

__all__ = (
    "EvaluationCaseResult",
    "EvaluationSuite",
    "EvaluationSuiteDefinition",
    "EvaluationSuiteStorage",
    "SuiteEvaluatorCaseResult",
    "SuiteEvaluatorResult",
    "evaluation_suite",
)


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
    results: Sequence[ScenarioEvaluatorResult] = Field(
        description="Evaluation results",
    )
    meta: Meta = Field(
        description="Additional evaluation metadata",
        default=META_EMPTY,
    )

    @property
    def passed(self) -> bool:
        # empty results is equivalent of failure
        return len(self.results) > 0 and all(result.passed for result in self.results)

    def report(
        self,
        *,
        include_passed: bool = True,
        include_details: bool = True,
    ) -> str:
        report: str = "\n".join(
            [
                result.report(
                    include_passed=include_passed,
                    include_details=include_details,
                )
                for result in self.results
                if include_passed or not result.passed
            ]
        )

        if report:  # nonempty report
            if include_details:
                return (
                    f"<evaluation_case identifier='{self.case.identifier}'>"
                    f"\n<relative_score>{self.relative_score * 100:.2f}%</relative_score>"
                    f"\n<evaluated_value>{self.value}</evaluated_value>"
                    # TODO: convert DataModel to xml representation when avaialble
                    f"\n<parameters>{self.case.parameters}</parameters>"
                    f"\n{report}"
                    "\n</evaluation_case>"
                )

            else:
                return f"Evaluation case {self.case.identifier}:\n{report}\n---\n"

        elif not self.results:
            return f"Evaluation case {self.case.identifier} empty!"

        else:
            return f"Evaluation case {self.case.identifier} passed!"

    @property
    def relative_score(self) -> float:
        if not self.results:
            return 0.0

        score: float = 0.0
        for evaluation in self.results:
            score += evaluation.relative_score

        return score / len(self.results)


class SuiteEvaluatorResult[CaseParameters: DataModel, Value: DataModel | str](DataModel):
    cases: Sequence[SuiteEvaluatorCaseResult[CaseParameters, Value]]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    def report(
        self,
        *,
        include_passed: bool = True,
        include_details: bool = True,
    ) -> str:
        report: str = "\n".join(
            [
                result.report(
                    include_passed=include_passed,
                    include_details=include_details,
                )
                for result in self.cases
                if include_passed or not result.passed
            ]
        )

        if report:  # nonempty report contains failing reports
            return f"Evaluation suite failed:\n\n{report}"

        elif not self.cases:
            return "Evaluation suite empty!"

        else:
            return "Evaluation suite passed!"

    @property
    def relative_score(self) -> float:
        if not self.cases:
            return 0.0

        score: float = 0.0
        for evaluation in self.cases:
            score += evaluation.relative_score

        return score / len(self.cases)


class EvaluationCaseResult[Value: DataModel | str](DataModel):
    @classmethod
    def of(
        cls,
        results: ScenarioEvaluatorResult | EvaluatorResult,
        *_results: ScenarioEvaluatorResult | EvaluatorResult,
        value: Value,
        meta: Meta | MetaValues | None = None,
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
                    scenario="EvaluationSuite",
                    evaluations=tuple(free_results),
                )
            )

        return cls(
            value=value,
            results=tuple(scenario_results),
            meta=Meta.of(meta),
        )

    @classmethod
    async def evaluating(
        cls,
        value: Value,
        /,
        evaluators: PreparedScenarioEvaluator[Value] | PreparedEvaluator[Value],
        *_evaluators: PreparedScenarioEvaluator[Value] | PreparedEvaluator[Value],
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls.of(
            *await gather(
                *[evaluator(value) for evaluator in [evaluators, *_evaluators]],
                return_exceptions=False,
            ),
            value=value,
            meta=Meta.of(meta),
        )

    value: Value = Field(
        description="Evaluated value",
    )
    results: Sequence[ScenarioEvaluatorResult] = Field(
        description="Evaluation results",
    )
    meta: Meta = Field(
        description="Additional evaluation metadata",
        default=META_EMPTY,
    )


@runtime_checkable
class EvaluationSuiteDefinition[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
    Value: DataModel | str,
](Protocol):
    async def __call__(
        self,
        parameters: SuiteParameters,
        case_parameters: CaseParameters,
    ) -> EvaluationCaseResult[Value]: ...


class EvaluationSuiteData[SuiteParameters: DataModel, CaseParameters: DataModel](DataModel):
    parameters: SuiteParameters
    cases: Sequence[EvaluationSuiteCase[CaseParameters]] = Field(default_factory=tuple)


@runtime_checkable
class EvaluationSuiteStorage[SuiteParameters: DataModel, CaseParameters: DataModel](Protocol):
    async def load(
        self,
    ) -> EvaluationSuiteData[SuiteParameters, CaseParameters]: ...

    async def save(
        self,
        data: EvaluationSuiteData[SuiteParameters, CaseParameters],
    ) -> None: ...


class EvaluationSuite[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
    Value: DataModel | str,
]:
    __slots__ = (
        "_case_parameters",
        "_data_cache",
        "_definition",
        "_execution_context",
        "_lock",
        "_storage",
        "_suite_parameters",
    )

    def __init__(
        self,
        suite_parameters: type[SuiteParameters],
        case_parameters: type[CaseParameters],
        definition: EvaluationSuiteDefinition[SuiteParameters, CaseParameters, Value],
        storage: EvaluationSuiteStorage[SuiteParameters, CaseParameters],
        execution_context: ScopeContext | None,
    ) -> None:
        self._suite_parameters: type[SuiteParameters] = suite_parameters
        self._case_parameters: type[CaseParameters] = case_parameters
        self._definition: EvaluationSuiteDefinition[SuiteParameters, CaseParameters, Value] = (
            definition
        )
        self._storage: EvaluationSuiteStorage[SuiteParameters, CaseParameters] = storage
        self._execution_context: ScopeContext | None = execution_context
        self._data_cache: EvaluationSuiteData[SuiteParameters, CaseParameters] | None = None
        self._lock: Lock = Lock()

    @overload
    async def __call__(
        self,
        parameters: SuiteParameters,
        /,
        *,
        case_parameters: CaseParameters,
    ) -> SuiteEvaluatorCaseResult[CaseParameters, Value]: ...

    @overload
    async def __call__(
        self,
        parameters: SuiteParameters,
        /,
        *,
        case_parameters: UUID,
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
        parameters: SuiteParameters | None = None,
        /,
        *,
        case_parameters: CaseParameters | UUID | None = None,
        reload: bool = False,
    ) -> (
        SuiteEvaluatorResult[CaseParameters, Value]
        | SuiteEvaluatorCaseResult[CaseParameters, Value]
    ):
        if context := self._execution_context:
            async with context:
                return await self._evaluate(
                    case_parameters,
                    suite_parameters=parameters,
                    reload=reload,
                )

        else:
            async with ctx.scope("evaluation.suite"):
                return await self._evaluate(
                    case_parameters,
                    suite_parameters=parameters,
                    reload=reload,
                )

    async def _evaluate(
        self,
        case_parameters: CaseParameters | UUID | None = None,
        /,
        *,
        suite_parameters: SuiteParameters | None = None,
        reload: bool = False,
    ) -> (
        SuiteEvaluatorResult[CaseParameters, Value]
        | SuiteEvaluatorCaseResult[CaseParameters, Value]
    ):
        suite_data: EvaluationSuiteData[SuiteParameters, CaseParameters] = await self._data(
            reload=reload
        )
        async with self._lock:
            match case_parameters:
                case None:
                    return SuiteEvaluatorResult(
                        cases=await gather(
                            *[
                                self._evaluate_case(
                                    case,
                                    suite_parameters=suite_parameters or suite_data.parameters,
                                )
                                for case in suite_data.cases
                            ],
                            return_exceptions=False,
                        )
                    )

                case UUID() as identifier:
                    if evaluation_case := next(
                        iter([case for case in suite_data.cases if case.identifier == identifier]),
                        None,
                    ):
                        return await self._evaluate_case(
                            evaluation_case,
                            suite_parameters=suite_parameters or suite_data.parameters,
                        )

                    else:
                        raise ValueError(f"Evaluation case with ID {identifier} does not exists.")

                case case_parameters:
                    return await self._evaluate_case(
                        EvaluationSuiteCase[CaseParameters](
                            parameters=case_parameters,
                        ),
                        suite_parameters=suite_parameters or suite_data.parameters,
                    )

    async def _evaluate_case(
        self,
        case_parameters: EvaluationSuiteCase[CaseParameters],
        *,
        suite_parameters: SuiteParameters,
    ) -> SuiteEvaluatorCaseResult[CaseParameters, Value]:
        result: EvaluationCaseResult[Value] = await self._definition(
            parameters=suite_parameters,
            case_parameters=case_parameters.parameters,
        )

        return SuiteEvaluatorCaseResult[CaseParameters, Value](
            case=case_parameters,
            value=result.value,
            results=result.results,
        )

    async def _data(
        self,
        reload: bool = False,
    ) -> EvaluationSuiteData[SuiteParameters, CaseParameters]:
        if reload or self._data_cache is None:
            self._data_cache = await self._storage.load()
            return self._data_cache

        else:
            return self._data_cache

    def with_execution_context(
        self,
        context: ScopeContext,
        /,
    ) -> Self:
        return self.__class__(
            suite_parameters=self._suite_parameters,
            case_parameters=self._case_parameters,
            definition=self._definition,
            storage=self._storage,
            execution_context=context,
        )

    def with_storage(
        self,
        storage: EvaluationSuiteStorage[SuiteParameters, CaseParameters]
        | EvaluationSuiteData[SuiteParameters, CaseParameters]
        | Path
        | str,
        /,
    ) -> Self:
        suite_storage: EvaluationSuiteStorage[SuiteParameters, CaseParameters]
        match storage:
            case str() as path_str:
                suite_storage = _EvaluationSuiteFileStorage[SuiteParameters, CaseParameters](
                    path=path_str,
                    data_type=EvaluationSuiteData[self._suite_parameters, self._case_parameters],
                )

            case Path() as path:
                suite_storage = _EvaluationSuiteFileStorage[SuiteParameters, CaseParameters](
                    path=path,
                    data_type=EvaluationSuiteData[self._suite_parameters, self._case_parameters],
                )

            case EvaluationSuiteData() as data:
                suite_storage = _EvaluationSuiteMemoryStorage[SuiteParameters, CaseParameters](
                    data_type=EvaluationSuiteData[SuiteParameters, CaseParameters],
                    parameters=data.parameters,
                    cases=data.cases,
                )

            case storage:
                suite_storage = storage

        return self.__class__(
            suite_parameters=self._suite_parameters,
            case_parameters=self._case_parameters,
            definition=self._definition,
            storage=suite_storage,
            execution_context=self._execution_context,
        )

    async def cases(
        self,
        reload: bool = False,
    ) -> Sequence[EvaluationSuiteCase[CaseParameters]]:
        async with self._lock:
            return (await self._data(reload=reload)).cases

    async def add_case(
        self,
        parameters: CaseParameters,
        /,
        comment: str | None = None,
    ) -> None:
        async with self._lock:
            data: EvaluationSuiteData[SuiteParameters, CaseParameters] = await self._data(
                reload=True
            )
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
        count: int,
        persist: bool = False,
        guidelines: str | None = None,
        examples: Iterable[CaseParameters] | None = None,
    ) -> Sequence[EvaluationSuiteCase[CaseParameters]]:
        async with self._lock:
            current: EvaluationSuiteData[SuiteParameters, CaseParameters] = await self._data(
                reload=True
            )

            generated: Sequence[EvaluationSuiteCase[CaseParameters]] = [
                EvaluationSuiteCase(parameters=parameters)
                for parameters in await generate_case_parameters(
                    self._case_parameters,
                    count=count,
                    examples=examples or [case.parameters for case in current.cases],
                    guidelines=guidelines,
                )
            ]

            self._data_cache = current.updated(cases=(*current.cases, *generated))

            if persist:
                await self._storage.save(self._data_cache)

            return generated

    async def remove_case(
        self,
        identifier: UUID,
        /,
    ) -> None:
        async with self._lock:
            data: EvaluationSuiteData[SuiteParameters, CaseParameters] = await self._data(
                reload=True
            )
            self._data_cache = data.updated(
                cases=tuple(case for case in data.cases if case.identifier != identifier)
            )
            await self._storage.save(self._data_cache)


def evaluation_suite[SuiteParameters: DataModel, CaseParameters: DataModel, Value: DataModel | str](
    case_parameters: type[CaseParameters],
    /,
    suite_parameters: type[SuiteParameters] | SuiteParameters,
    *,
    storage: EvaluationSuiteStorage[SuiteParameters, CaseParameters]
    | EvaluationSuiteData[SuiteParameters, CaseParameters]
    | Path
    | str
    | None = None,
    execution_context: ScopeContext | None = None,
) -> Callable[
    [EvaluationSuiteDefinition[SuiteParameters, CaseParameters, Value]],
    EvaluationSuite[SuiteParameters, CaseParameters, Value],
]:
    suite_parameters_type: type[SuiteParameters]
    suite_parameters_value: SuiteParameters | None
    if isinstance(suite_parameters, type):
        suite_parameters_type = suite_parameters
        suite_parameters_value = None

    else:
        suite_parameters_type = type(suite_parameters)
        suite_parameters_value = suite_parameters

    suite_storage: EvaluationSuiteStorage[SuiteParameters, CaseParameters]
    match storage:
        case None:
            if suite_parameters_value is None:
                try:
                    # try to prepare default value if able
                    suite_parameters_value = suite_parameters_type()

                except Exception:
                    raise RuntimeError(
                        "Failed to create default suite parameters."
                        " Can't define evaluation suite without parameters and storage."
                    ) from None

            suite_storage = _EvaluationSuiteMemoryStorage[SuiteParameters, CaseParameters](
                data_type=EvaluationSuiteData[suite_parameters_type, case_parameters],
                parameters=suite_parameters_value,
                cases=(),
            )

        case str() as path_str:
            suite_storage = _EvaluationSuiteFileStorage[SuiteParameters, CaseParameters](
                path=path_str,
                data_type=EvaluationSuiteData[suite_parameters, case_parameters],
            )

        case Path() as path:
            suite_storage = _EvaluationSuiteFileStorage[SuiteParameters, CaseParameters](
                path=path,
                data_type=EvaluationSuiteData[suite_parameters, case_parameters],
            )

        case EvaluationSuiteData() as data:
            suite_storage = _EvaluationSuiteMemoryStorage[SuiteParameters, CaseParameters](
                data_type=EvaluationSuiteData[suite_parameters, case_parameters],
                parameters=data.parameters,
                cases=data.cases,
            )

        case storage:
            suite_storage = storage

    def wrap(
        definition: EvaluationSuiteDefinition[SuiteParameters, CaseParameters, Value],
    ) -> EvaluationSuite[SuiteParameters, CaseParameters, Value]:
        return EvaluationSuite[SuiteParameters, CaseParameters, Value](
            suite_parameters=suite_parameters_type,
            case_parameters=case_parameters,
            definition=definition,
            storage=suite_storage,
            execution_context=execution_context,
        )

    return wrap


class _EvaluationSuiteMemoryStorage[SuiteParameters: DataModel, CaseParameters: DataModel]:
    __slots__ = (
        "_cases_store",
        "_data_type",
        "_parameters_store",
    )

    def __init__(
        self,
        data_type: type[EvaluationSuiteData[SuiteParameters, CaseParameters]],
        parameters: SuiteParameters,
        cases: Sequence[EvaluationSuiteCase[CaseParameters]] | None,
    ) -> None:
        self._parameters_store: SuiteParameters = parameters
        self._cases_store: Sequence[EvaluationSuiteCase[CaseParameters]] = cases or ()
        self._data_type: type[EvaluationSuiteData[SuiteParameters, CaseParameters]] = data_type

    async def load(
        self,
    ) -> EvaluationSuiteData[SuiteParameters, CaseParameters]:
        return self._data_type(
            parameters=self._parameters_store,
            cases=self._cases_store,
        )

    async def save(
        self,
        data: EvaluationSuiteData[SuiteParameters, CaseParameters],
    ) -> None:
        self._cases_store = data.cases


class _EvaluationSuiteFileStorage[SuiteParameters: DataModel, CaseParameters: DataModel]:
    __slots__ = (
        "_data_type",
        "_path",
    )

    def __init__(
        self,
        path: Path | str,
        data_type: type[EvaluationSuiteData[SuiteParameters, CaseParameters]],
    ) -> None:
        self._path: Path
        match path:
            case str() as path_str:
                self._path = Path(path_str)

            case path:
                self._path = path

        self._data_type: type[EvaluationSuiteData[SuiteParameters, CaseParameters]] = data_type

    async def load(
        self,
    ) -> EvaluationSuiteData[SuiteParameters, CaseParameters]:
        try:
            return await self._file_load(data_type=self._data_type)

        except ValueError as exc:
            ctx.log_error(
                f"Invalid EvaluationSuite at {self._path}",
                exception=exc,
            )
            raise exc

    @asynchronous
    def _file_load(
        self,
        data_type: type[EvaluationSuiteData[SuiteParameters, CaseParameters]],
    ) -> EvaluationSuiteData[SuiteParameters, CaseParameters]:
        if self._path.exists():
            with open(self._path, mode="rb") as file:
                return data_type.from_json(file.read())

        else:
            raise RuntimeError(f"Missing evaluation suite data at {self._path}")

    async def save(
        self,
        data: EvaluationSuiteData[SuiteParameters, CaseParameters],
    ) -> None:
        await self._file_save(data=data)

    @asynchronous
    def _file_save(
        self,
        data: EvaluationSuiteData[SuiteParameters, CaseParameters],
    ) -> None:
        with open(self._path, mode="wb+") as file:
            file.write(data.to_json(indent=2).encode("utf-8"))
