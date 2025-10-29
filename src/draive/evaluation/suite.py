import json
import random
from asyncio import Lock
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Annotated, Protocol, Self, cast, runtime_checkable
from uuid import uuid4

from haiway import (
    Default,
    Description,
    Immutable,
    Meta,
    MetaValues,
    State,
    asynchronous,
    ctx,
    execute_concurrently,
)

from draive.evaluation.evaluator import EvaluatorResult
from draive.evaluation.scenario import EvaluatorScenarioResult
from draive.parameters import DataModel

__all__ = (
    "EvaluatorSuite",
    "EvaluatorSuiteCaseResult",
    "EvaluatorSuiteCasesStorage",
    "EvaluatorSuiteDefinition",
    "EvaluatorSuiteResult",
    "PreparedEvaluatorSuite",
    "evaluator_suite",
)


class EvaluatorSuiteCase[Parameters: DataModel](DataModel):
    """
    Individual test case for an evaluator suite.

    Represents a single test case with parameters, unique identifier.

    Attributes
    ----------
    identifier : str
        Unique identifier for the test case
    parameters : CaseParameters
        Parameters specific to this test case
    """

    identifier: str = Default(default_factory=lambda: str(uuid4()))
    parameters: Parameters


class EvaluatorSuiteCaseResult(DataModel):
    """
    Result of evaluating a single test case.

    Contains the test case identifier and its evaluation results, with methods to
    check if the case passed and generate reports.

    Attributes
    ----------
    case : EvaluatorSuiteCase[CaseParameters]
        The test case that was evaluated
    results : Sequence[EvaluatorScenarioResult]
        Results from all scenario evaluations
    """

    case_identifier: Annotated[
        str,
        Description("Evaluated case identifier"),
    ]
    results: Annotated[
        Sequence[EvaluatorScenarioResult | EvaluatorResult],
        Description("Evaluation results"),
    ]

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
            return f"{self.case_identifier} has no evaluations"

        status: str = "passed" if self.passed else "failed"
        results_report: str = "\n".join(
            result.report(
                detailed=detailed,
                include_passed=include_passed,
            )
            if isinstance(result, EvaluatorScenarioResult)
            else result.report(
                detailed=detailed,
            )
            for result in self.results
            if include_passed or not result.passed
        )

        if not detailed:
            if results_report:
                return f"{self.case_identifier}: {status} ({self.performance}%)\n{results_report}"

            else:
                return f"{self.case_identifier}: {status} ({self.performance}%)"

        if results_report:
            return (
                f"<evaluator_suite case='{self.case_identifier}' status='{status}'"
                f" performance='{self.performance:.2f}%'>"
                f"\n<results>\n{results_report}\n</results>"
                "\n</evaluator_suite>"
            )

        else:
            return f"<evaluator_suite case='{self.case_identifier}' status='{status}' />"

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
        for result in self.results:
            score += min(100.0, result.performance)  # normalize results

        return score / len(self.results)


class EvaluatorSuiteResult(DataModel):
    """
    Aggregated results from evaluating an entire test suite.

    Contains results from all test cases evaluated in a suite, providing
    methods to check overall pass/fail status, calculate aggregate performance,
    and generate comprehensive reports.

    Attributes
    ----------
    suite : str
        Name of the evaluated suite
    results : Sequence[EvaluatorSuiteCaseResult]
        Results from all evaluated test cases
    """

    suite: str
    results: Sequence[EvaluatorSuiteCaseResult]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def report(
        self,
        *,
        detailed: bool = True,
        include_passed: bool = True,
    ) -> str:
        if not self.results:
            return "Evaluator suite empty!"

        status: str = "passed" if self.passed else "failed"
        results_report: str = "\n---\n".join(
            result.report(
                detailed=detailed,
                include_passed=include_passed,
            )
            for result in self.results
            if include_passed or not result.passed
        )

        if not detailed:
            if results_report:
                return (
                    f"Evaluator suite {self.suite}: {status} ({self.performance:.2f}%):"
                    f"\n\n{results_report}"
                )

            else:
                return f"Evaluator suite {self.suite}: {status} ({self.performance:.2f}%)"

        if results_report:
            return (
                f"<evaluator_suite name='{self.suite}' status='{status}'"
                f" performance='{self.performance:.2f}%'>"
                f"\n<results>\n{results_report}\n</results>"
                "\n</evaluator_suite>"
            )

        else:
            return (
                f"<evaluator_suite name='{self.suite}' status='{status}'"
                f" performance='{self.performance:.2f}%' />"
            )

    @property
    def performance(self) -> float:
        if not self.results:
            return 0.0

        score: float = 0.0
        for result in self.results:
            score += min(100.0, result.performance)  # normalize results

        return score / len(self.results)


@runtime_checkable
class EvaluatorSuiteDefinition[**Args, Parameters: DataModel](Protocol):
    """
    Protocol for evaluator suite function definitions.

    Defines the interface for functions that can evaluate test cases within
    a suite. Suite definitions receive case parameters and additional arguments,
    returning a collection of evaluation results (either scenario results or
    individual evaluator results).

    The protocol requires a __name__ property for identification and a callable
    interface that processes parameters and returns evaluation results.
    """

    @property
    def __name__(self) -> str: ...

    async def __call__(
        self,
        parameters: Parameters,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Sequence[EvaluatorScenarioResult | EvaluatorResult]: ...


@runtime_checkable
class EvaluatorSuiteCasesStorage[Parameters: DataModel](Protocol):
    """
    Protocol for test case storage backends.

    Defines the interface for storing and retrieving evaluation suite test cases.
    Implementations can provide different storage mechanisms (memory, file, database)
    while maintaining a consistent interface for case persistence.

    Methods
    -------
    load()
        Load all test cases from storage
    save(cases)
        Persist test cases to storage
    """

    async def load(
        self,
    ) -> Sequence[EvaluatorSuiteCase[Parameters]]: ...

    async def save(
        self,
        cases: Sequence[EvaluatorSuiteCase[Parameters]],
    ) -> None: ...


@runtime_checkable
class PreparedEvaluatorSuite[Parameters: DataModel](Protocol):
    """
    Protocol for evaluator suites with pre-configured arguments.

    A prepared evaluator suite has all arguments except the test cases already
    bound, simplifying the evaluation interface. This allows for easy reuse of
    suite configurations with different case selections.

    The callable interface accepts various case selection strategies:
    - None: Run all available cases
    - int: Run a specific number of random cases
    - float: Run a percentage of available cases (0.0-1.0)
    - Sequence: Run specific cases by ID, parameters, or case objects
    """

    async def __call__(
        self,
        cases: Sequence[EvaluatorSuiteCase[Parameters] | Parameters | str]
        | float
        | int
        | None = None,
        /,
    ) -> EvaluatorSuiteResult: ...


class EvaluatorSuite[**Args, Parameters: DataModel](Immutable):
    """
    Comprehensive test suite for systematic evaluation.

    An EvaluatorSuite manages a collection of test cases and coordinates their
    evaluation using a defined evaluation function. It provides case management
    (add, remove, generate), flexible case selection strategies, concurrent
    evaluation support, and pluggable storage backends.

    The suite execution uses "evaluator.suite.suite_name" context where suite_name
    is the actual name of the evaluator suite.

    Attributes
    ----------
    name : str
        Identifier for the suite
    concurrent_evaluations : int
        Maximum number of test cases to evaluate concurrently
    meta : Meta
        Additional metadata for the suite

    Examples
    --------
    >>> @evaluator_suite(MyParameters, name="validation_suite")
    ... async def validate(params: MyParameters) -> Sequence[EvaluatorResult]:
    ...     return [await evaluator1(params), await evaluator2(params)]
    ...
    >>> # Run all cases
    >>> results = await validate()
    ...
    >>> # Run 5 random cases
    >>> results = await validate(5)
    ...
    >>> # Run 50% of cases
    >>> results = await validate(0.5)
    """

    name: str
    concurrent_evaluations: int
    meta: Meta
    _parameters: type[Parameters]
    _definition: EvaluatorSuiteDefinition[Args, Parameters]
    _cases_storage: EvaluatorSuiteCasesStorage[Parameters]
    _cases_cache: Sequence[EvaluatorSuiteCase[Parameters]] | None
    _state: Sequence[State]
    _lock: Lock

    def __init__(
        self,
        name: str,
        concurrent_evaluations: int,
        parameters: type[Parameters],
        definition: EvaluatorSuiteDefinition[Args, Parameters],
        cases_storage: EvaluatorSuiteCasesStorage[Parameters],
        state: Sequence[State],
        meta: Meta,
    ) -> None:
        object.__setattr__(
            self,
            "name",
            name,
        )
        object.__setattr__(
            self,
            "concurrent_evaluations",
            concurrent_evaluations,
        )
        object.__setattr__(
            self,
            "meta",
            meta,
        )
        object.__setattr__(
            self,
            "_parameters",
            parameters,
        )
        object.__setattr__(
            self,
            "_definition",
            definition,
        )
        object.__setattr__(
            self,
            "_cases_storage",
            cases_storage,
        )
        object.__setattr__(
            self,
            "_cases_cache",
            None,
        )
        object.__setattr__(
            self,
            "_state",
            state,
        )
        object.__setattr__(
            self,
            "_lock",
            Lock(),
        )

    def prepared(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> PreparedEvaluatorSuite[Parameters]:
        async def evaluate(
            cases: Sequence[EvaluatorSuiteCase[Parameters] | Parameters | str]
            | float
            | int
            | None = None,
            /,
        ) -> EvaluatorSuiteResult:
            return await self(
                cases,
                *args,
                **kwargs,
            )

        return evaluate

    async def __call__(
        self,
        cases: Sequence[EvaluatorSuiteCase[Parameters] | Parameters | str]
        | float
        | int
        | None = None,
        /,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluatorSuiteResult:
        async with ctx.scope(f"evaluator.suite.{self.name}", *self._state):
            available_cases: Sequence[EvaluatorSuiteCase[Parameters]]
            async with self._lock:
                available_cases = await self._available_cases()

            selected_cases: Sequence[EvaluatorSuiteCase[Parameters]]
            if cases is None:
                selected_cases = available_cases

            elif isinstance(cases, int):
                selected_cases = random.sample(  # nosec: B311
                    available_cases,
                    min(len(available_cases), cases),
                )

            elif isinstance(cases, float):
                assert 0 < cases <= 1  # nosec: B101
                selected_cases = random.sample(  # nosec: B311
                    available_cases,
                    min(len(available_cases), int(len(available_cases) * cases)),
                )

            else:
                selected_cases = []
                for case in cases:
                    if isinstance(case, str):
                        if evaluation_case := next(
                            iter(
                                available
                                for available in available_cases
                                if available.identifier == case
                            ),
                            None,
                        ):
                            selected_cases.append(evaluation_case)

                        else:
                            raise ValueError(f"Evaluation case with ID {case} does not exists.")

                    elif isinstance(case, EvaluatorSuiteCase):
                        selected_cases.append(cast(EvaluatorSuiteCase[Parameters], case))

                    else:
                        selected_cases.append(EvaluatorSuiteCase(parameters=case))

            async def evaluate_case(
                case: EvaluatorSuiteCase[Parameters],
            ) -> EvaluatorSuiteCaseResult:
                return await self._evaluate_case(
                    case,
                    *args,
                    **kwargs,
                )

            return EvaluatorSuiteResult(
                suite=self.name,
                results=await execute_concurrently(
                    evaluate_case,
                    selected_cases,
                    concurrent_tasks=self.concurrent_evaluations,
                ),
            )

    async def _evaluate_case(
        self,
        case: EvaluatorSuiteCase[Parameters],
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> EvaluatorSuiteCaseResult:
        results: Sequence[EvaluatorScenarioResult | EvaluatorResult] = await self._definition(
            case.parameters,
            *args,
            **kwargs,
        )

        return EvaluatorSuiteCaseResult(
            case_identifier=case.identifier,
            results=results,
        )

    async def _available_cases(
        self,
        reload: bool = False,
    ) -> Sequence[EvaluatorSuiteCase[Parameters]]:
        if reload or self._cases_cache is None:
            object.__setattr__(
                self,
                "_cases_cache",
                await self._cases_storage.load(),
            )

            assert self._cases_cache is not None  # nosec: B101
            return self._cases_cache

        else:
            return self._cases_cache

    def with_name(
        self,
        name: str,
        /,
    ) -> Self:
        return self.__class__(
            name=name,
            concurrent_evaluations=self.concurrent_evaluations,
            meta=self.meta,
            parameters=self._parameters,
            definition=self._definition,
            cases_storage=self._cases_storage,
            state=self._state,
        )

    def with_concurrent_evaluations(
        self,
        concurrent_evaluations: int,
        /,
    ) -> Self:
        return self.__class__(
            name=self.name,
            concurrent_evaluations=concurrent_evaluations,
            meta=self.meta,
            parameters=self._parameters,
            definition=self._definition,
            cases_storage=self._cases_storage,
            state=self._state,
        )

    def with_meta(
        self,
        meta: Meta | MetaValues,
        /,
    ) -> Self:
        return self.__class__(
            name=self.name,
            concurrent_evaluations=self.concurrent_evaluations,
            meta=self.meta.merged_with(meta),
            parameters=self._parameters,
            definition=self._definition,
            cases_storage=self._cases_storage,
            state=self._state,
        )

    def with_state(
        self,
        state: State,
        /,
        *states: State,
    ) -> Self:
        return self.__class__(
            name=self.name,
            concurrent_evaluations=self.concurrent_evaluations,
            meta=self.meta,
            parameters=self._parameters,
            definition=self._definition,
            cases_storage=self._cases_storage,
            state=(*self._state, state, *states),
        )

    def with_storage(
        self,
        storage: EvaluatorSuiteCasesStorage[Parameters]
        | Sequence[EvaluatorSuiteCase[Parameters]]
        | Path
        | str,
        /,
    ) -> Self:
        suite_storage: EvaluatorSuiteCasesStorage[Parameters]
        if isinstance(storage, Path | str):
            suite_storage = _EvaluatorSuiteFileStorage[Parameters](
                path=storage,
                parameters=self._parameters,
            )

        elif isinstance(storage, EvaluatorSuiteCasesStorage):
            suite_storage = storage

        else:
            suite_storage = _EvaluatorSuiteMemoryStorage[Parameters](
                cases=storage,
            )

        return self.__class__(
            name=self.name,
            concurrent_evaluations=self.concurrent_evaluations,
            meta=self.meta,
            parameters=self._parameters,
            definition=self._definition,
            cases_storage=suite_storage,
            state=self._state,
        )

    async def cases(
        self,
        *,
        reload: bool = False,
    ) -> Sequence[EvaluatorSuiteCase[Parameters]]:
        async with self._lock:
            return await self._available_cases(reload=reload)

    async def add_case(
        self,
        parameters: Parameters,
        /,
        identifier: str | None = None,
    ) -> None:
        async with self._lock:
            current_cases: Sequence[EvaluatorSuiteCase[Parameters]] = await self._available_cases(
                reload=True
            )
            object.__setattr__(
                self,
                "_cases_cache",
                (
                    *current_cases,
                    EvaluatorSuiteCase[Parameters](
                        identifier=identifier if identifier is not None else str(uuid4()),
                        parameters=parameters,
                    ),
                ),
            )

            assert self._cases_cache is not None  # nosec: B101
            await self._cases_storage.save(self._cases_cache)

    async def generate_cases(
        self,
        *,
        count: int,
        persist: bool = False,
        guidelines: str | None = None,
        examples: Iterable[Parameters] | None = None,
    ) -> Sequence[EvaluatorSuiteCase[Parameters]]:
        async with self._lock:
            from draive.helpers.evaluation_case_generation import generate_case_parameters

            current_cases: Sequence[EvaluatorSuiteCase[Parameters]] = await self._available_cases(
                reload=True
            )

            generated: Sequence[EvaluatorSuiteCase[Parameters]] = [
                EvaluatorSuiteCase(parameters=parameters)
                for parameters in await generate_case_parameters(
                    self._parameters,
                    count=count,
                    examples=examples or [case.parameters for case in current_cases],
                    guidelines=guidelines,
                )
            ]

            object.__setattr__(
                self,
                "_cases_cache",
                (*current_cases, *generated),
            )

            assert self._cases_cache is not None  # nosec: B101
            if persist:
                await self._cases_storage.save(self._cases_cache)

            return generated

    async def remove_case(
        self,
        identifier: str,
        /,
    ) -> None:
        async with self._lock:
            current_cases: Sequence[EvaluatorSuiteCase[Parameters]] = await self._available_cases(
                reload=True
            )
            object.__setattr__(
                self,
                "_cases_cache",
                tuple(case for case in current_cases if case.identifier != identifier),
            )
            assert self._cases_cache is not None  # nosec: B101
            await self._cases_storage.save(self._cases_cache)


def evaluator_suite[**Args, Parameters: DataModel](
    parameters: type[Parameters],
    /,
    *,
    name: str | None = None,
    storage: EvaluatorSuiteCasesStorage[Parameters]
    | Sequence[EvaluatorSuiteCase[Parameters]]
    | Path
    | str
    | None = None,
    concurrent_evaluations: int = 2,
    state: Sequence[State] = (),
    meta: Meta | MetaValues | None = None,
) -> Callable[
    [EvaluatorSuiteDefinition[Args, Parameters]],
    EvaluatorSuite[Args, Parameters],
]:
    """
    Create or decorate an evaluator suite function.

    Returns a decorator that transforms a suite definition function into an
    EvaluatorSuite instance with case management and evaluation capabilities.

    Parameters
    ----------
    parameters : type[Parameters]
        Type of parameters for test cases in the suite
    name : str | None, optional
        Name for the suite. If None, uses function's __name__
    storage : EvaluatorSuiteCasesStorage | Sequence | Path | str | None, optional
        Storage backend for test cases. Can be:
        - None: In-memory storage with empty initial cases
        - Path/str: File-based storage at specified path
        - Sequence[EvaluatorSuiteCase]: In-memory storage with initial cases
        - EvaluatorSuiteCasesStorage: Custom storage backend
    concurrent_evaluations : int, optional
        Maximum concurrent test case evaluations, by default 2
    state : Sequence[State], optional
        State objects to include in evaluation context
    meta : Meta | MetaValues | None, optional
        Metadata for the suite

    Returns
    -------
    Callable[[EvaluatorSuiteDefinition[Args, Parameters]], EvaluatorSuite[Args, Parameters]]
        Decorator that creates an EvaluatorSuite from a definition function

    Examples
    --------
    >>> @evaluator_suite(
    ...     TestParams,
    ...     name="integration_tests",
    ...     storage="./test_cases.json",
    ...     concurrent_evaluations=5
    ... )
    ... async def integration_suite(params: TestParams) -> Sequence[EvaluatorResult]:
    ...     return [
    ...         await api_test(params.api_endpoint),
    ...         await db_test(params.db_config)
    ...     ]
    ...
    >>> # Add test cases
    >>> await integration_suite.add_case(TestParams(api_endpoint="..."))
    ...
    >>> # Run all test cases
    >>> results = await integration_suite()
    """
    suite_storage: EvaluatorSuiteCasesStorage[Parameters]
    if storage is None:
        suite_storage = _EvaluatorSuiteMemoryStorage[Parameters](
            cases=(),
        )

    elif isinstance(storage, Path | str):
        suite_storage = _EvaluatorSuiteFileStorage[Parameters](
            path=storage,
            parameters=parameters,
        )

    elif isinstance(storage, EvaluatorSuiteCasesStorage):
        suite_storage = storage

    else:
        suite_storage = _EvaluatorSuiteMemoryStorage[Parameters](
            cases=storage,
        )

    def wrap(
        definition: EvaluatorSuiteDefinition[Args, Parameters],
    ) -> EvaluatorSuite[Args, Parameters]:
        return EvaluatorSuite[Args, Parameters](
            name=name or definition.__name__,
            concurrent_evaluations=concurrent_evaluations,
            parameters=parameters,
            definition=definition,
            cases_storage=suite_storage,
            state=state,
            meta=Meta.of(meta),
        )

    return wrap


class EvaluatorSuiteData[Parameters: DataModel](DataModel):
    cases: Sequence[EvaluatorSuiteCase[Parameters]] = ()


class _EvaluatorSuiteMemoryStorage[Parameters: DataModel](Immutable):
    _cases_store: Sequence[EvaluatorSuiteCase[Parameters]]

    def __init__(
        self,
        cases: Sequence[EvaluatorSuiteCase[Parameters]] | None,
    ) -> None:
        object.__setattr__(
            self,
            "_cases_store",
            cases or (),
        )

    async def load(
        self,
    ) -> Sequence[EvaluatorSuiteCase[Parameters]]:
        return self._cases_store

    async def save(
        self,
        cases: Sequence[EvaluatorSuiteCase[Parameters]],
    ) -> None:
        object.__setattr__(
            self,
            "_cases_store",
            cases,
        )


class _EvaluatorSuiteFileStorage[Parameters: DataModel](Immutable):
    _path: Path
    _data_type: type[EvaluatorSuiteCase[Parameters]]

    def __init__(
        self,
        path: Path | str,
        parameters: type[Parameters],
    ) -> None:
        match path:
            case str() as path_str:
                object.__setattr__(
                    self,
                    "_path",
                    Path(path_str),
                )

            case path:
                object.__setattr__(
                    self,
                    "_path",
                    path,
                )

        object.__setattr__(
            self,
            "_data_type",
            EvaluatorSuiteCase[parameters],
        )

    async def load(
        self,
    ) -> Sequence[EvaluatorSuiteCase[Parameters]]:
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
        data_type: type[EvaluatorSuiteCase[Parameters]],
    ) -> Sequence[EvaluatorSuiteCase[Parameters]]:
        if self._path.exists():
            with open(self._path, mode="rb") as file:
                match json.loads(file.read()):
                    case [*cases]:
                        return tuple(data_type.from_mapping(case) for case in cases)

                    case _:
                        raise RuntimeError(f"Invalid evaluation suite data at {self._path}")

        else:
            raise RuntimeError(f"Missing evaluation suite data at {self._path}")

    async def save(
        self,
        cases: Sequence[EvaluatorSuiteCase[Parameters]],
    ) -> None:
        await self._file_save(cases)

    @asynchronous
    def _file_save(
        self,
        cases: Sequence[EvaluatorSuiteCase[Parameters]],
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, mode="wb+") as file:
            file.write(json.dumps([case.to_mapping() for case in cases]).encode("utf-8"))
