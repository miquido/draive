from asyncio import (
    FIRST_COMPLETED,
    Future,
    InvalidStateError,
    get_running_loop,
    wait,
)
from collections.abc import Callable, Coroutine
from contextvars import ContextVar, Token
from typing import Any, ParamSpec, Protocol, Self, TypeVar

from draive.metrics import Metric
from draive.scope import ctx
from draive.types import State

__all__ = [
    "allowing_early_exit",
    "with_early_exit",
]

_Args_T = ParamSpec("_Args_T")
_Result_T = TypeVar("_Result_T")
_EarlyExitResult_T = TypeVar(
    "_EarlyExitResult_T",
    bound=State,
)


async def allowing_early_exit(
    result: type[_EarlyExitResult_T],
    call: Callable[_Args_T, Coroutine[Any, Any, _Result_T]],
    /,
    *args: _Args_T.args,
    **kwargs: _Args_T.kwargs,
) -> _Result_T | _EarlyExitResult_T:
    early_exit_future: Future[_EarlyExitResult_T] = get_running_loop().create_future()

    async def exit_early(early_result: _Result_T) -> None:
        try:
            if not isinstance(early_result, result):
                raise TypeError(
                    "Early exit result not matching expected",
                    result,
                    type(early_result),
                )
            early_exit_future.set_result(early_result)
            ctx.record(_EarlyExitResultTrace.of(early_result))
        except InvalidStateError as exc:
            ctx.log_debug("Ignored redundant attempt to early exit: %s", exc)
        except TypeError as exc:
            ctx.log_debug("Ignored attempt to early exit with unexpected result: %s", exc)

    early_exit_token: Token[_RequestEarlyExit] = _EarlyExit_Var.set(exit_early)
    try:
        finished, running = await wait(
            [
                ctx.spawn_task(call, *args, **kwargs),
                early_exit_future,
            ],
            return_when=FIRST_COMPLETED,
        )

        for task in running:  # pyright: ignore[reportUnknownVariableType]
            task.cancel()

        return finished.pop().result()

    finally:
        _EarlyExit_Var.reset(early_exit_token)


async def with_early_exit(result: _EarlyExitResult_T) -> _EarlyExitResult_T:
    try:
        await _EarlyExit_Var.get()(early_result=result)
    except LookupError as exc:
        ctx.log_debug("Requested early exit in context not allowing it: %s", exc)
    return result


class _RequestEarlyExit(Protocol):
    async def __call__(
        self,
        early_result: Any,
    ) -> None: ...


class _EarlyExitResultTrace(Metric):
    @classmethod
    def of(
        cls,
        value: Any,
        /,
    ) -> Self:
        return cls(result=value)

    result: Any


_EarlyExit_Var = ContextVar[_RequestEarlyExit]("_EarlyExit_Var")
