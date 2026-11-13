from asyncio import Task, TaskGroup
from collections.abc import Coroutine, Sequence

__all__ = ("gather_all",)


async def gather_all[Result](
    *coros: Coroutine[None, None, Result],
) -> Sequence[Result]:
    """Await all given awaitables concurrently, cancelling the rest when one fails.

    Unlike ``asyncio.gather`` the remaining awaitables do not keep running in
    the background after the first failure.

    Parameters
    ----------
    *coros : Coroutine[None, None, Result]
        Coroutines to run concurrently.

    Returns
    -------
    Sequence[Result]
        Results in the order of the given coroutines.

    Raises
    ------
    Exception
        The first exception raised by any of the awaitables.
    """

    tasks: Sequence[Task[Result]]
    try:
        async with TaskGroup() as group:
            tasks = tuple(group.create_task(awaitable) for awaitable in coros)

    except ExceptionGroup as group_exc:
        # surface the original failure instead of the group wrapper
        raise group_exc.exceptions[0] from None

    return tuple(task.result() for task in tasks)
