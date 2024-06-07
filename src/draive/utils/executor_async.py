from asyncio import get_running_loop, iscoroutinefunction
from collections.abc import Callable, Coroutine

from draive.utils.mimic import mimic_function

__all__ = [
    "run_async",
]


def run_async[**Args, Result](
    function: Callable[Args, Result],
    /,
) -> Callable[Args, Coroutine[None, None, Result]]:
    """\
    Simple wrapper for a sync function to run in loop executor.
    The result is an async function.

    Parameters
    ----------
    function: Callable[Args, Result]
        function to be wrapped as running in loop executor

    Returns
    -------
    Callable[_Args, _Result]
        function wrapped to async using loop executor
    """

    assert not iscoroutinefunction(function), "Cannot wrap async function in executor"  # nosec: B101

    return _ExecutorWrapper(function)


class _ExecutorWrapper[**Args, Result]:
    def __init__(
        self,
        function: Callable[Args, Result],
        /,
    ) -> None:
        self._function: Callable[Args, Result] = function
        # TODO: prepare function converting kwargs to args
        # taking into account defaults

        # mimic function attributes if able
        mimic_function(function, within=self)

    async def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        assert not kwargs, "Referring to argument keys is not supported yet"  # nosec: B101

        return await get_running_loop().run_in_executor(  # pyright: ignore[reportUnknownVariableType]
            None,
            self._function,
            # not implemented by pyright - https://github.com/microsoft/pyright/discussions/5049
            *args,  # pyright: ignore[reportCallIssue]
        )
