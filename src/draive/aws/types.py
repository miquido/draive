from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable

from haiway import MQQueue, ObservabilityAttribute

__all__ = (
    "AWSAccessDenied",
    "AWSCloudwatchEventPutting",
    "AWSCloudwatchLogPutting",
    "AWSCloudwatchMetricPutting",
    "AWSError",
    "AWSResourceNotFound",
    "AWSSQSQueueAccessing",
)


class AWSError(Exception):
    """Base error raised when an AWS request fails.

    Parameters
    ----------
    uri
        URI of the AWS resource involved in the failing request.
    code
        AWS error code when available.
    message
        Human readable error message returned by AWS.
    """

    __slots__ = (
        "code",
        "message",
        "uri",
    )

    def __init__(
        self,
        *,
        uri: str,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        detail: str = message or "AWS request failed"
        if code:
            detail = f"{detail} ({code})"

        super().__init__(f"{detail} - {uri}")
        self.uri: str = uri
        self.code: str | None = code
        self.message: str | None = message


class AWSAccessDenied(AWSError):
    """Error raised when AWS denies access to the requested resource."""

    def __init__(
        self,
        *,
        uri: str,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            code=code or "AccessDenied",
            message=message or "Access denied",
        )


class AWSResourceNotFound(AWSError):
    """Error raised when the requested AWS resource cannot be found."""

    def __init__(
        self,
        *,
        uri: str,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            code=code or "ResourceNotFound",
            message=message or "Resource not found",
        )


@runtime_checkable
class AWSSQSQueueAccessing(Protocol):
    async def __call__[Content](
        self,
        queue: str,
        content_encoder: Callable[[Content], str],
        content_decoder: Callable[[str], Content],
        **extra: Any,
    ) -> AbstractAsyncContextManager[MQQueue[Content]]: ...


@runtime_checkable
class AWSCloudwatchLogPutting(Protocol):
    async def __call__(
        self,
        *,
        log_group: str,
        log_stream: str,
        message: str,
    ) -> None: ...


@runtime_checkable
class AWSCloudwatchMetricPutting(Protocol):
    async def __call__(
        self,
        *,
        namespace: str,
        metric: str,
        value: float | int,
        unit: str | None,
        attributes: Mapping[str, ObservabilityAttribute],
    ) -> None: ...


@runtime_checkable
class AWSCloudwatchEventPutting(Protocol):
    async def __call__(
        self,
        *,
        event_bus: str,
        event_source: str,
        detail_type: str,
        detail: str,
    ) -> None: ...
