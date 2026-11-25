import asyncio

__all__ = ("completed",)


async def completed[Result](
    task: asyncio.Task[Result],
    /,
) -> Result:
    # cleanup has to complete however the caller was cancelled, also by scopes which cancel
    # again every await - the cancellation is propagated once the task is done
    cancelled: bool = False
    while not task.done():
        try:
            await asyncio.shield(task)

        except asyncio.CancelledError:
            cancelled = True

    if cancelled:
        raise asyncio.CancelledError

    return task.result()
