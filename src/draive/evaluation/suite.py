from asyncio import Lock, gather
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, overload, runtime_checkable
from uuid import UUID, uuid4

from draive.evaluation.scenario import ScenarioEvaluatorResult
from draive.parameters import DataModel, Field
from draive.scope import ctx
from draive.types import frozenlist
from draive.utils import asynchronous

__all__ = [
    "evaluation_suite",
    "EvaluationCaseResult",
    "EvaluationSuite",
    "EvaluationSuiteCaseResult",
    "EvaluationSuiteDefinition",
    "EvaluationSuiteStorage",
]


class EvaluationSuiteCase[CaseParameters: DataModel](DataModel):
    identifier: UUID = Field(default_factory=uuid4)
    parameters: CaseParameters
    comment: str | None = None


class EvaluationSuiteCaseResult[CaseParameters: DataModel, Value: DataModel | str](DataModel):
    case: EvaluationSuiteCase[CaseParameters] = Field(
        description="Evaluated case",
    )
    value: Value = Field(
        description="Evaluated value",
    )
    results: frozenlist[ScenarioEvaluatorResult] = Field(
        description="Evaluation results",
    )

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


class EvaluationCaseResult[Value: DataModel | str](DataModel):
    value: Value = Field(
        description="Evaluated value",
    )
    results: frozenlist[ScenarioEvaluatorResult] = Field(
        description="Evaluation results",
    )


@runtime_checkable
class EvaluationSuiteDefinition[CaseParameters: DataModel, Value: DataModel | str](Protocol):
    async def __call__(
        self,
        evaluation_case: CaseParameters,
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
    ) -> None:
        self._definition: EvaluationSuiteDefinition[CaseParameters, Value] = definition
        self._storage: EvaluationSuiteStorage[CaseParameters] = storage
        self._data_cache: EvaluationSuiteData[CaseParameters] | None = None
        self._lock: Lock = Lock()

    @overload
    async def __call__(
        self,
        *,
        evaluated_case: CaseParameters | UUID | None,
        reload: bool = False,
    ) -> EvaluationSuiteCaseResult[CaseParameters, Value]: ...

    @overload
    async def __call__(
        self,
        *,
        reload: bool = False,
    ) -> list[EvaluationSuiteCaseResult[CaseParameters, Value]]: ...

    async def __call__(
        self,
        *,
        evaluated_case: CaseParameters | UUID | None = None,
        reload: bool = False,
    ) -> (
        list[EvaluationSuiteCaseResult[CaseParameters, Value]]
        | EvaluationSuiteCaseResult[CaseParameters, Value]
    ):
        async with self._lock:
            match evaluated_case:
                case None:
                    return await gather(
                        *[
                            self._evaluate(evaluated_case=case)
                            for case in (await self._data(reload=reload)).cases
                        ],
                        return_exceptions=False,
                    )

                case UUID() as identifier:
                    available_cases: frozenlist[EvaluationSuiteCase[CaseParameters]] = (
                        await self._data(reload=reload)
                    ).cases

                    if evaluation_case := next(
                        iter([case for case in available_cases if case.identifier == identifier]),
                        None,
                    ):
                        return await self._evaluate(evaluated_case=evaluation_case)

                    else:
                        raise ValueError(f"Evaluation case with ID {identifier} does not exists.")

                case case_parameters:
                    return await self._evaluate(
                        evaluated_case=EvaluationSuiteCase[CaseParameters](
                            parameters=case_parameters,
                        )
                    )

    async def _evaluate(
        self,
        *,
        evaluated_case: EvaluationSuiteCase[CaseParameters],
    ) -> EvaluationSuiteCaseResult[CaseParameters, Value]:
        case_result: EvaluationCaseResult[Value] = await self._definition(
            evaluation_case=evaluated_case.parameters
        )

        return EvaluationSuiteCaseResult[CaseParameters, Value](
            case=evaluated_case,
            value=case_result.value,
            results=case_result.results,
        )

    async def _data(
        self,
        reload: bool = False,
    ) -> EvaluationSuiteData[CaseParameters]:
        if (data := self._data_cache) and not reload:
            return data

        else:
            self._data_cache = await self._storage.load()
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

    @asynchronous(executor=None)
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

    @asynchronous(executor=None)
    def _file_save(
        self,
        data: EvaluationSuiteData[CaseParameters],
    ) -> None:
        with open(self._path, mode="wb+") as file:
            file.write(data.as_json(indent=2).encode("utf-8"))
