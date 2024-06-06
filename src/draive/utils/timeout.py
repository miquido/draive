from asyncio import AbstractEventLoop, Future, Task, TimerHandle, get_running_loop
from collections.abc import Callable, Coroutine

from draive.utils.mimic import mimic_function

__all__ = [
    "with_timeout",
]


def with_timeout[**Args, Result](
    timeout: float,
    /,
) -> Callable[
    [Callable[Args, Coroutine[None, None, Result]]],
    Callable[Args, Coroutine[None, None, Result]],
]:
    """\
    Simple timeout wrapper for the function call. \
    When the timeout time will pass before function returns it will be \
    cancelled and TimeoutError exception will raise. Make sure that wrapped \
    function handles cancellation properly.
    This wrapper is not thread safe.

    Parameters
    ----------
    timeout: float
        timeout time in seconds

    Returns
    -------
    Callable[[Callable[_Args, _Result]], Callable[_Args, _Result]] | Callable[_Args, _Result]
        function wrapper adding timeout
    """

    def _wrap(
        function: Callable[Args, Coroutine[None, None, Result]],
    ) -> Callable[Args, Coroutine[None, None, Result]]:
        return _AsyncTimeout(
            function,
            timeout=timeout,
        )

    return _wrap


class _AsyncTimeout[**Args, Result]:
    def __init__(
        self,
        function: Callable[Args, Coroutine[None, None, Result]],
        /,
        timeout: float,
    ) -> None:
        self._function: Callable[Args, Coroutine[None, None, Result]] = function
        self._timeout: float = timeout

        # mimic function attributes if able
        mimic_function(function, within=self)

    async def __call__(
        self,
        *args: Args.args,
        **kwargs: Args.kwargs,
    ) -> Result:
        loop: AbstractEventLoop = get_running_loop()
        future: Future[Result] = loop.create_future()
        task: Task[Result] = loop.create_task(
            self._function(
                *args,
                **kwargs,
            ),
        )

        def on_timeout(
            future: Future[Result],
        ) -> None:
            if future.done():
                return  # ignore if already finished

            # result future on its completion will ensure that task will complete
            future.set_exception(TimeoutError())

        timeout_handle: TimerHandle = loop.call_later(
            self._timeout,
            on_timeout,
            future,
        )

        def on_completion(
            task: Task[Result],
        ) -> None:
            timeout_handle.cancel()  # at this stage we no longer need timeout to trigger

            if future.done():
                return  # ignore if already finished

            try:
                future.set_result(task.result())

            except Exception as exc:
                future.set_exception(exc)

        task.add_done_callback(on_completion)

        def on_result(
            future: Future[Result],
        ) -> None:
            task.cancel()  # when result future completes make sure that task also completes

        future.add_done_callback(on_result)

        return await future
