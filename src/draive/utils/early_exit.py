from asyncio import (
    FIRST_COMPLETED,
    Future,
    InvalidStateError,
    get_running_loop,
    wait,
)
from collections.abc import Callable, Coroutine
from contextvars import ContextVar, Token
from typing import Any, Protocol, Self

from draive.scope import ctx
from draive.types import Model

__all__ = [
    "allowing_early_exit",
    "with_early_exit",
]


async def allowing_early_exit[**Args, Result, EarlyResult](
    result: type[EarlyResult],
    call: Callable[Args, Coroutine[Any, Any, Result]],
    /,
    *args: Args.args,
    **kwargs: Args.kwargs,
) -> Result | EarlyResult:
    early_exit_future: Future[EarlyResult] = get_running_loop().create_future()

    async def exit_early(early_result: EarlyResult) -> None:
        if not isinstance(early_result, result):
            return ctx.log_debug(
                "Ignored attempt to early exit with unexpected result: %s",
                type(early_result),
            )

        try:
            early_exit_future.set_result(early_result)
            ctx.record(_EarlyExitResultTrace.of(early_result))
        except InvalidStateError as exc:
            ctx.log_debug("Ignored redundant attempt to early exit: %s", exc)

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


async def with_early_exit[Result](result: Result) -> Result:
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


class _EarlyExitResultTrace(Model):
    @classmethod
    def of(
        cls,
        value: Any,
        /,
    ) -> Self:
        return cls(result=value)

    result: Any


_EarlyExit_Var = ContextVar[_RequestEarlyExit]("_EarlyExit_Var")
