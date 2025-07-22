import random
from asyncio import Lock
from collections.abc import Callable, Collection, Iterable, Sequence
from pathlib import Path
from typing import Protocol, Self, runtime_checkable
from uuid import UUID, uuid4

from haiway import (
    META_EMPTY,
    Meta,
    MetaValues,
    State,
    as_list,
    asynchronous,
    ctx,
    execute_concurrently,
)

from draive.evaluation.evaluator import EvaluatorResult, PreparedEvaluator
from draive.evaluation.generator import generate_case_parameters
from draive.evaluation.scenario import EvaluatorScenarioResult, PreparedEvaluatorScenario
from draive.parameters import DataModel, Field

__all__ = (
    "EvaluatorCaseResult",
    "EvaluatorSuite",
    "EvaluatorSuiteCaseResult",
    "EvaluatorSuiteDefinition",
    "EvaluatorSuiteResult",
    "EvaluatorSuiteStorage",
    "evaluator_suite",
)


class EvaluatorSuiteCase[CaseParameters: DataModel](DataModel):
    """
    Individual test case for an evaluator suite.

    Represents a single test case with parameters, unique identifier,
    and optional comment for documentation.

    Attributes
    ----------
    identifier : str
        Unique identifier for the test case
    parameters : CaseParameters
        Parameters specific to this test case
    comment : str | None
        Optional comment describing the test case
    """

    identifier: str = Field(default_factory=lambda: str(uuid4()))
    parameters: CaseParameters
    comment: str | None = None


class EvaluatorSuiteCaseResult[CaseParameters: DataModel](DataModel):
    """
    Result of evaluating a single test case.

    Contains the test case and its evaluation results, with methods to
    check if the case passed and generate reports.

    Attributes
    ----------
    case : EvaluatorSuiteCase[CaseParameters]
        The test case that was evaluated
    results : Sequence[EvaluatorScenarioResult]
        Results from all scenario evaluations
    meta : Meta
        Additional metadata for the case result
    """

    case: EvaluatorSuiteCase[CaseParameters] = Field(
        description="Evaluated case",
    )
    results: Sequence[EvaluatorScenarioResult] = Field(
        description="Evaluation results",
    )
    meta: Meta = Field(
        description="Additional evaluation metadata",
        default=META_EMPTY,
    )

    @property
    def passed(self) -> bool:
        """
        Check if all scenario evaluations passed.

        Returns
        -------
        bool
            True if all scenario evaluations passed, False otherwise.
            Empty results return False.
        """
        # empty results is equivalent of failure
        return len(self.results) > 0 and all(result.passed for result in self.results)

    def report(
        self,
        *,
        detailed: bool = True,
        include_passed: bool = True,
    ) -> str:
        """
        Generate a human-readable report of the case results.

        Parameters
        ----------
        detailed : bool, optional
            If True, include full details in XML format. If False, return
            brief summary, by default True
        include_passed : bool, optional
            If True, include passed evaluations in the report. If False,
            only show failed evaluations, by default True

        Returns
        -------
        str
            Formatted report string
        """
        if not self.results:
            return f"{self.case.identifier} has no evaluations"

        status: str = "passed" if self.passed else "failed"
        results_report: str = "\n".join(
            result.report(
                detailed=detailed,
                include_passed=include_passed,
            )
            for result in self.results
            if include_passed or not result.passed
        )

        if not detailed:
            if results_report:
                return f"{self.case.identifier}: {status} ({self.performance}%)\n{results_report}"

            else:
                return f"{self.case.identifier}: {status} ({self.performance}%)"

        if results_report:
            return (
                f"<evaluator_suite case='{self.case.identifier}' status='{status}'"
                f" performance='{self.performance:.2f}%'>"
                f"\n<results>\n{results_report}\n</results>"
                "\n</evaluator_suite>"
            )

        else:
            return f"<evaluator_suite case='{self.case.identifier}' status='{status}' />"

    @property
    def performance(self) -> float:
        """
        Calculate average performance across all scenario evaluations.

        Returns
        -------
        float
            Average performance percentage (0-100) across all evaluations.
            Returns 0.0 if no evaluations.
        """
        if not self.results:
            return 0.0

        score: float = 0.0
        for evaluation in self.results:
            score += evaluation.performance

        return score / len(self.results)


class EvaluatorSuiteResult[SuiteParameters: DataModel, CaseParameters: DataModel](DataModel):
    parameters: SuiteParameters
    cases: Sequence[EvaluatorSuiteCaseResult[CaseParameters]]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    def report(
        self,
        *,
        detailed: bool = True,
        include_passed: bool = True,
    ) -> str:
        if not self.cases:
            return "Evaluator suite empty!"

        status: str = "passed" if self.passed else "failed"
        results_report: str = "\n---\n".join(
            result.report(
                detailed=detailed,
                include_passed=include_passed,
            )
            for result in self.cases
            if include_passed or not result.passed
        )

        if not detailed:
            if results_report:
                return f"Evaluator suite: {status} ({self.performance:.2f}%):\n\n{results_report}"

            else:
                return f"Evaluator suite: {status} ({self.performance:.2f}%)"

        if results_report:
            return (
                f"<evaluator_suite status='{status}'"
                f" performance='{self.performance:.2f}%'>"
                f"\n<results>\n{results_report}\n</results>"
                "\n</evaluator_suite>"
            )

        else:
            return f"<evaluator_suite status='{status}' performance='{self.performance:.2f}%' />"

    @property
    def performance(self) -> float:
        if not self.cases:
            return 0.0

        score: float = 0.0
        for evaluation in self.cases:
            score += evaluation.performance

        return score / len(self.cases)


class EvaluatorCaseResult(DataModel):
    @classmethod
    def of(
        cls,
        result: EvaluatorScenarioResult | EvaluatorResult,
        *results: EvaluatorScenarioResult | EvaluatorResult,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        free_results: list[EvaluatorResult] = []
        scenario_results: list[EvaluatorScenarioResult] = []
        for element in (result, *results):
            if isinstance(element, EvaluatorScenarioResult):
                scenario_results.append(element)

            else:
                free_results.append(element)

        if free_results:
            scenario_results.append(
                EvaluatorScenarioResult(
                    scenario="EvaluatorSuite",
                    evaluations=free_results,
                )
            )

        return cls(
            results=scenario_results,
            meta=Meta.of(meta),
        )

    @classmethod
    async def evaluating[Value](
        cls,
        value: Value,
        /,
        evaluator: PreparedEvaluatorScenario[Value] | PreparedEvaluator[Value],
        *evaluators: PreparedEvaluatorScenario[Value] | PreparedEvaluator[Value],
        concurrent_tasks: int = 2,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        async def execute(
            evaluator: PreparedEvaluatorScenario[Value] | PreparedEvaluator[Value],
        ) -> EvaluatorScenarioResult | EvaluatorResult:
            return await evaluator(value)

        return cls.of(
            *await execute_concurrently(
                execute,
                (evaluator, *evaluators),
                concurrent_tasks=concurrent_tasks,
            ),
            meta=Meta.of(meta),
        )

    @classmethod
    def merging(
        cls,
        result: Self,
        *results: Self,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        merged_evaluations: list[EvaluatorScenarioResult] = as_list(result.results)
        merged_meta: Meta = result.meta
        for other in results:
            merged_evaluations.extend(other.results)
            merged_meta = merged_meta.merged_with(other.meta)

        return cls(
            results=merged_evaluations,
            meta=merged_meta.merged_with(Meta.of(meta)),
        )

    results: Sequence[EvaluatorScenarioResult] = Field(
        description="Evaluation results",
    )
    meta: Meta = Field(
        description="Additional evaluation metadata",
        default=META_EMPTY,
    )


@runtime_checkable
class EvaluatorSuiteDefinition[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
](Protocol):
    async def __call__(
        self,
        parameters: SuiteParameters,
        case_parameters: CaseParameters,
    ) -> EvaluatorCaseResult: ...


class EvaluatorSuiteData[SuiteParameters: DataModel, CaseParameters: DataModel](DataModel):
    parameters: SuiteParameters
    cases: Sequence[EvaluatorSuiteCase[CaseParameters]] = ()


@runtime_checkable
class EvaluatorSuiteStorage[SuiteParameters: DataModel, CaseParameters: DataModel](Protocol):
    async def load(
        self,
    ) -> EvaluatorSuiteData[SuiteParameters, CaseParameters]: ...

    async def save(
        self,
        data: EvaluatorSuiteData[SuiteParameters, CaseParameters],
    ) -> None: ...


class EvaluatorSuite[
    SuiteParameters: DataModel,
    CaseParameters: DataModel,
]:
    __slots__ = (
        "_case_parameters",
        "_data_cache",
        "_definition",
        "_lock",
        "_state",
        "_storage",
        "_suite_parameters",
    )

    def __init__(
        self,
        suite_parameters: type[SuiteParameters],
        case_parameters: type[CaseParameters],
        definition: EvaluatorSuiteDefinition[SuiteParameters, CaseParameters],
        storage: EvaluatorSuiteStorage[SuiteParameters, CaseParameters],
        state: Collection[State],
    ) -> None:
        self._suite_parameters: type[SuiteParameters] = suite_parameters
        self._case_parameters: type[CaseParameters] = case_parameters
        self._definition: EvaluatorSuiteDefinition[SuiteParameters, CaseParameters] = definition
        self._storage: EvaluatorSuiteStorage[SuiteParameters, CaseParameters] = storage
        self._state: Collection[State] = state
        self._data_cache: EvaluatorSuiteData[SuiteParameters, CaseParameters] | None = None
        self._lock: Lock = Lock()

    async def __call__(
        self,
        case_parameters: Sequence[EvaluatorSuiteCase[CaseParameters] | CaseParameters | UUID]
        | float
        | int
        | None = None,
        /,
        parameters: SuiteParameters | None = None,
        concurrent_cases: int = 2,
        reload: bool = False,
    ) -> EvaluatorSuiteResult[SuiteParameters, CaseParameters]:
        async with ctx.scope("evaluation.suite", *self._state):
            return await self._evaluate(
                case_parameters,
                suite_parameters=parameters,
                concurrent_cases=concurrent_cases,
                reload=reload,
            )

    async def _evaluate(
        self,
        case_parameters: Sequence[EvaluatorSuiteCase[CaseParameters] | CaseParameters | UUID]
        | float
        | int
        | None,
        /,
        *,
        suite_parameters: SuiteParameters | None,
        concurrent_cases: int,
        reload: bool,
    ) -> EvaluatorSuiteResult[SuiteParameters, CaseParameters]:
        suite_data: EvaluatorSuiteData[SuiteParameters, CaseParameters]
        async with self._lock:
            suite_data = await self._data(reload=reload)

        cases: Sequence[EvaluatorSuiteCase[CaseParameters]] = []
        if not case_parameters:
            cases = suite_data.cases

        elif isinstance(case_parameters, int):
            cases = random.sample(  # nosec: B311
                suite_data.cases,
                min(len(suite_data.cases), case_parameters),
            )

        elif isinstance(case_parameters, float):
            assert 0 < case_parameters <= 1  # nosec: B101
            cases = random.sample(  # nosec: B311
                suite_data.cases,
                min(len(suite_data.cases), int(len(suite_data.cases) * case_parameters)),
            )

        else:
            cases = []
            for case in case_parameters:
                match case:
                    case UUID() as case_id:
                        if evaluation_case := next(
                            iter(case for case in suite_data.cases if case.identifier == case_id),
                            None,
                        ):
                            cases.append(evaluation_case)

                        else:
                            raise ValueError(f"Evaluation case with ID {case_id} does not exists.")

                    case EvaluatorSuiteCase():
                        cases.append(case)

                    case _:
                        cases.append(EvaluatorSuiteCase[CaseParameters](parameters=case))

        parameters: SuiteParameters = suite_parameters or suite_data.parameters

        async def evaluate_case(
            case: EvaluatorSuiteCase[CaseParameters],
        ) -> EvaluatorSuiteCaseResult[CaseParameters]:
            return await self._evaluate_case(
                case,
                suite_parameters=parameters,
            )

        return EvaluatorSuiteResult(
            parameters=parameters,
            cases=await execute_concurrently(
                evaluate_case,
                cases,
                concurrent_tasks=concurrent_cases,
            ),
        )

    async def _evaluate_case(
        self,
        case_parameters: EvaluatorSuiteCase[CaseParameters],
        *,
        suite_parameters: SuiteParameters,
    ) -> EvaluatorSuiteCaseResult[CaseParameters]:
        result: EvaluatorCaseResult = await self._definition(
            parameters=suite_parameters,
            case_parameters=case_parameters.parameters,
        )

        return EvaluatorSuiteCaseResult[CaseParameters](
            case=case_parameters,
            results=result.results,
        )

    async def _data(
        self,
        reload: bool = False,
    ) -> EvaluatorSuiteData[SuiteParameters, CaseParameters]:
        if reload or self._data_cache is None:
            self._data_cache = await self._storage.load()
            return self._data_cache

        else:
            return self._data_cache

    def with_state(
        self,
        state: State,
        /,
        *states: State,
    ) -> Self:
        return self.__class__(
            suite_parameters=self._suite_parameters,
            case_parameters=self._case_parameters,
            definition=self._definition,
            storage=self._storage,
            state=(*self._state, state, *states),
        )

    def with_storage(
        self,
        storage: EvaluatorSuiteStorage[SuiteParameters, CaseParameters]
        | EvaluatorSuiteData[SuiteParameters, CaseParameters]
        | Path
        | str,
        /,
    ) -> Self:
        suite_storage: EvaluatorSuiteStorage[SuiteParameters, CaseParameters]
        match storage:
            case str() as path_str:
                suite_storage = _EvaluatorSuiteFileStorage[SuiteParameters, CaseParameters](
                    path=path_str,
                    data_type=EvaluatorSuiteData[self._suite_parameters, self._case_parameters],
                )

            case Path() as path:
                suite_storage = _EvaluatorSuiteFileStorage[SuiteParameters, CaseParameters](
                    path=path,
                    data_type=EvaluatorSuiteData[self._suite_parameters, self._case_parameters],
                )

            case EvaluatorSuiteData() as data:
                suite_storage = _EvaluatorSuiteMemoryStorage[SuiteParameters, CaseParameters](
                    data_type=EvaluatorSuiteData[SuiteParameters, CaseParameters],
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
            state=self._state,
        )

    async def cases(
        self,
        reload: bool = False,
    ) -> Sequence[EvaluatorSuiteCase[CaseParameters]]:
        async with self._lock:
            return (await self._data(reload=reload)).cases

    async def add_case(
        self,
        parameters: CaseParameters,
        /,
        comment: str | None = None,
    ) -> None:
        async with self._lock:
            data: EvaluatorSuiteData[SuiteParameters, CaseParameters] = await self._data(
                reload=True
            )
            self._data_cache = data.updated(
                cases=(
                    *data.cases,
                    EvaluatorSuiteCase[CaseParameters](
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
    ) -> Sequence[EvaluatorSuiteCase[CaseParameters]]:
        async with self._lock:
            current: EvaluatorSuiteData[SuiteParameters, CaseParameters] = await self._data(
                reload=True
            )

            generated: Sequence[EvaluatorSuiteCase[CaseParameters]] = [
                EvaluatorSuiteCase(parameters=parameters)
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
            data: EvaluatorSuiteData[SuiteParameters, CaseParameters] = await self._data(
                reload=True
            )
            self._data_cache = data.updated(
                cases=tuple(case for case in data.cases if case.identifier != identifier)
            )
            await self._storage.save(self._data_cache)


def evaluator_suite[SuiteParameters: DataModel, CaseParameters: DataModel, Value](
    case_parameters: type[CaseParameters],
    /,
    suite_parameters: type[SuiteParameters] | SuiteParameters,
    *,
    storage: EvaluatorSuiteStorage[SuiteParameters, CaseParameters]
    | EvaluatorSuiteData[SuiteParameters, CaseParameters]
    | Path
    | str
    | None = None,
    state: Collection[State] = (),
) -> Callable[
    [EvaluatorSuiteDefinition[SuiteParameters, CaseParameters]],
    EvaluatorSuite[SuiteParameters, CaseParameters],
]:
    suite_parameters_type: type[SuiteParameters]
    suite_parameters_value: SuiteParameters | None
    if isinstance(suite_parameters, type):
        suite_parameters_type = suite_parameters
        suite_parameters_value = None

    else:
        suite_parameters_type = type(suite_parameters)
        suite_parameters_value = suite_parameters

    suite_storage: EvaluatorSuiteStorage[SuiteParameters, CaseParameters]
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

            suite_storage = _EvaluatorSuiteMemoryStorage[SuiteParameters, CaseParameters](
                data_type=EvaluatorSuiteData[suite_parameters_type, case_parameters],
                parameters=suite_parameters_value,
                cases=(),
            )

        case str() as path_str:
            suite_storage = _EvaluatorSuiteFileStorage[SuiteParameters, CaseParameters](
                path=path_str,
                data_type=EvaluatorSuiteData[suite_parameters, case_parameters],
            )

        case Path() as path:
            suite_storage = _EvaluatorSuiteFileStorage[SuiteParameters, CaseParameters](
                path=path,
                data_type=EvaluatorSuiteData[suite_parameters, case_parameters],
            )

        case EvaluatorSuiteData() as data:
            suite_storage = _EvaluatorSuiteMemoryStorage[SuiteParameters, CaseParameters](
                data_type=EvaluatorSuiteData[suite_parameters, case_parameters],
                parameters=data.parameters,
                cases=data.cases,
            )

        case storage:
            suite_storage = storage

    def wrap(
        definition: EvaluatorSuiteDefinition[SuiteParameters, CaseParameters],
    ) -> EvaluatorSuite[SuiteParameters, CaseParameters]:
        return EvaluatorSuite[SuiteParameters, CaseParameters](
            suite_parameters=suite_parameters_type,
            case_parameters=case_parameters,
            definition=definition,
            storage=suite_storage,
            state=state,
        )

    return wrap


class _EvaluatorSuiteMemoryStorage[SuiteParameters: DataModel, CaseParameters: DataModel]:
    __slots__ = (
        "_cases_store",
        "_data_type",
        "_parameters_store",
    )

    def __init__(
        self,
        data_type: type[EvaluatorSuiteData[SuiteParameters, CaseParameters]],
        parameters: SuiteParameters,
        cases: Sequence[EvaluatorSuiteCase[CaseParameters]] | None,
    ) -> None:
        self._parameters_store: SuiteParameters = parameters
        self._cases_store: Sequence[EvaluatorSuiteCase[CaseParameters]] = cases or ()
        self._data_type: type[EvaluatorSuiteData[SuiteParameters, CaseParameters]] = data_type

    async def load(
        self,
    ) -> EvaluatorSuiteData[SuiteParameters, CaseParameters]:
        return self._data_type(
            parameters=self._parameters_store,
            cases=self._cases_store,
        )

    async def save(
        self,
        data: EvaluatorSuiteData[SuiteParameters, CaseParameters],
    ) -> None:
        self._cases_store = data.cases


class _EvaluatorSuiteFileStorage[SuiteParameters: DataModel, CaseParameters: DataModel]:
    __slots__ = (
        "_data_type",
        "_path",
    )

    def __init__(
        self,
        path: Path | str,
        data_type: type[EvaluatorSuiteData[SuiteParameters, CaseParameters]],
    ) -> None:
        self._path: Path
        match path:
            case str() as path_str:
                self._path = Path(path_str)

            case path:
                self._path = path

        self._data_type: type[EvaluatorSuiteData[SuiteParameters, CaseParameters]] = data_type

    async def load(
        self,
    ) -> EvaluatorSuiteData[SuiteParameters, CaseParameters]:
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
        data_type: type[EvaluatorSuiteData[SuiteParameters, CaseParameters]],
    ) -> EvaluatorSuiteData[SuiteParameters, CaseParameters]:
        if self._path.exists():
            with open(self._path, mode="rb") as file:
                return data_type.from_json(file.read())

        else:
            raise RuntimeError(f"Missing evaluation suite data at {self._path}")

    async def save(
        self,
        data: EvaluatorSuiteData[SuiteParameters, CaseParameters],
    ) -> None:
        await self._file_save(data=data)

    @asynchronous
    def _file_save(
        self,
        data: EvaluatorSuiteData[SuiteParameters, CaseParameters],
    ) -> None:
        with open(self._path, mode="wb+") as file:
            file.write(data.to_json(indent=2).encode("utf-8"))
