from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, overload

from haiway import MQQueue, State, statemethod

from draive.aws.types import AWSSQSQueueAccessing

__all__ = ("AWSSQS",)


class AWSSQS(State):
    @overload
    @classmethod
    async def queue[Content](
        cls,
        queue: str,
        /,
        content_encoder: Callable[[Content], str],
        content_decoder: Callable[[str], Content],
        **extra: Any,
    ) -> AbstractAsyncContextManager[MQQueue[Content]]: ...

    @overload
    async def queue[Content](
        self,
        queue: str,
        /,
        content_encoder: Callable[[Content], str],
        content_decoder: Callable[[str], Content],
        **extra: Any,
    ) -> AbstractAsyncContextManager[MQQueue[Content]]: ...

    @statemethod
    async def queue[Content](
        self,
        queue: str,
        /,
        content_encoder: Callable[[Content], str],
        content_decoder: Callable[[str], Content],
        **extra: Any,
    ) -> AbstractAsyncContextManager[MQQueue[Content]]:
        """
        Acquire an async context manager for a typed AWS SQS queue bound to the current state.

        Parameters
        ----------
        queue : str
            Name of the queue to access on the broker.
        content_encoder : Callable[[Content], str]
            Callable that transforms typed payloads into broker-serializable objects before publish.
        content_decoder : Callable[[str], Content]
            Callable that converts received broker payloads into typed content for consumers.
        **extra : Any
            Additional options forwarded to the underlying queue accessor.

        Returns
        -------
        AbstractAsyncContextManager[MQQueue[Content]]
            Context manager yielding an `MQQueue` configured with the provided encoder/decoder.

        Notes
        -----
        Typical usage::

            async with AWSSQS.queue(
                "events",
                content_encoder=encode_event,
                content_decoder=decode_event,
            ) as queue:
                await queue.publish(event)
                async for message in await queue.consume():
                    ...

        The encoder is invoked for every publish and the decoder for every consumed payload.
        Entering the context establishes queue access and ensures clean teardown when
        the block exits.
        """
        return await self.queue_accessing(
            queue,
            content_encoder=content_encoder,
            content_decoder=content_decoder,
            **extra,
        )

    queue_accessing: AWSSQSQueueAccessing
