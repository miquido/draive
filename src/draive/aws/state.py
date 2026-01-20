from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, overload

from haiway import (
    MQQueue,
    ObservabilityAttribute,
    State,
    statemethod,
)

from draive.aws.types import (
    AWSCloudwatchEventPutting,
    AWSCloudwatchLogPutting,
    AWSCloudwatchMetricPutting,
    AWSSQSQueueAccessing,
)

__all__ = (
    "AWSSQS",
    "AWSCloudwatch",
)


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


class AWSCloudwatch(State):
    @overload
    @classmethod
    async def put_log(
        cls,
        *,
        log_group: str,
        log_stream: str,
        message: str,
    ) -> None: ...

    @overload
    async def put_log(
        self,
        *,
        log_group: str,
        log_stream: str,
        message: str,
    ) -> None: ...

    @statemethod
    async def put_log(
        self,
        *,
        log_group: str,
        log_stream: str,
        message: str,
    ) -> None:
        """
        Write a single log entry to CloudWatch Logs.

        Parameters
        ----------
        log_group
            CloudWatch Logs group name.
        log_stream
            CloudWatch Logs stream name.
        message
            Log message payload.
        """
        await self.log_putting(
            log_group=log_group,
            log_stream=log_stream,
            message=message,
        )

    @overload
    @classmethod
    async def put_metric(
        cls,
        *,
        namespace: str,
        metric: str,
        value: float | int,
        unit: str | None = None,
        attributes: Mapping[str, ObservabilityAttribute] | None = None,
    ) -> None: ...

    @overload
    async def put_metric(
        self,
        *,
        namespace: str,
        metric: str,
        value: float | int,
        unit: str | None = None,
        attributes: Mapping[str, ObservabilityAttribute] | None = None,
    ) -> None: ...

    @statemethod
    async def put_metric(
        self,
        *,
        namespace: str,
        metric: str,
        value: float | int,
        unit: str | None = None,
        attributes: Mapping[str, ObservabilityAttribute] | None = None,
    ) -> None:
        """
        Publish a single metric data point to CloudWatch.

        Parameters
        ----------
        namespace
            CloudWatch metrics namespace.
        metric
            Metric name.
        value
            Metric value.
        unit
            Optional unit name.
        attributes
            Dimension attributes for the metric.
        """
        await self.metric_putting(
            namespace=namespace,
            metric=metric,
            value=value,
            unit=unit,
            attributes=attributes or {},
        )

    @overload
    @classmethod
    async def put_event(
        cls,
        *,
        event_bus: str,
        event_source: str,
        detail_type: str,
        detail: str,
    ) -> None: ...

    @overload
    async def put_event(
        self,
        *,
        event_bus: str,
        event_source: str,
        detail_type: str,
        detail: str,
    ) -> None: ...

    @statemethod
    async def put_event(
        self,
        *,
        event_bus: str,
        event_source: str,
        detail_type: str,
        detail: str,
    ) -> None:
        """
        Publish an event to EventBridge.

        Parameters
        ----------
        event_bus
            EventBridge bus name.
        event_source
            Event source identifier.
        detail_type
            Event type name.
        detail
            JSON-serialized event payload.
        """
        await self.event_putting(
            event_bus=event_bus,
            event_source=event_source,
            detail_type=detail_type,
            detail=detail,
        )

    log_putting: AWSCloudwatchLogPutting
    metric_putting: AWSCloudwatchMetricPutting
    event_putting: AWSCloudwatchEventPutting
