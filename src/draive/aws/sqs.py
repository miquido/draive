from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from botocore.exceptions import ClientError  # pyright: ignore[reportMissingModuleSource]
from haiway import (
    META_EMPTY,
    FlatObject,
    MQMessage,
    MQQueue,
    RawValue,
    asynchronous,
    ctx,
)

from draive.aws.api import AWSAPI
from draive.aws.types import AWSAccessDenied, AWSError, AWSResourceNotFound

__all__ = ("AWSSQSMixin",)


class AWSSQSMixin(AWSAPI):
    def __init__(
        self,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        super().__init__(
            region_name=region_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            profile_name=profile_name,
        )

        self._queue_cache: MutableMapping[str, str] = {}

    async def _queue_access[Content](
        self,
        queue: str,
        content_encoder: Callable[[Content], str],
        content_decoder: Callable[[str], Content],
        **extra: Any,
    ) -> AbstractAsyncContextManager[MQQueue[Content]]:
        @asynccontextmanager
        async def context() -> AsyncGenerator[MQQueue[Content]]:
            async def publish_message(
                message: Content,
                attributes: FlatObject | None,
                **extra: Any,
            ) -> None:
                await self._publish(
                    content_encoder(message),
                    queue=queue,
                    attributes=attributes,
                )

            async def consume_messages(
                **extra: Any,
            ) -> AsyncIterable[MQMessage[Content]]:
                return self._consume(
                    queue,
                    decoder=content_decoder,
                )

            yield MQQueue[Content](
                publishing=publish_message,
                consuming=consume_messages,
            )

        return context()

    @asynchronous
    def _publish(
        self,
        body: str,
        /,
        *,
        queue: str,
        attributes: FlatObject | None = None,
    ) -> None:
        queue_url = self._resolve_queue_url(
            queue=queue,
            operation="send_message",
        )

        try:
            self._sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=body,
                MessageAttributes={
                    key: _format_attribute_value(value) for key, value in attributes.items()
                }
                if attributes
                else {},
            )

        except ClientError as exc:
            raise _translate_sqs_client_error(
                error=exc,
                queue=queue,
                operation="send_message",
            ) from exc

    async def _consume[Content](
        self,
        queue: str,
        /,
        *,
        decoder: Callable[[str], Content],
    ) -> AsyncIterator[MQMessage[Content]]:
        while True:  # Continue polling until cancelled
            ctx.check_cancellation()

            for message in await self._receive(queue=queue):
                receipt_handle: str = message["ReceiptHandle"]

                async def acknowledge(
                    receipt_handle: str = receipt_handle,
                    **extra: Any,
                ) -> None:
                    await self._ack_message(
                        queue=queue,
                        receipt_handle=receipt_handle,
                    )

                async def reject(
                    **extra: Any,
                ) -> None:
                    pass  # relying on automatic retries

                yield MQMessage(
                    content=decoder(message["Body"]),
                    acknowledge=acknowledge,
                    reject=reject,
                    meta=META_EMPTY,
                )

    @asynchronous
    def _receive(
        self,
        *,
        queue: str,
    ) -> Sequence[Mapping[str, Any]]:
        queue_url = self._resolve_queue_url(
            queue=queue,
            operation="receive_message",
        )

        try:
            return self._sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                MessageAttributeNames=["All"],
            ).get("Messages", [])

        except ClientError as exc:
            raise _translate_sqs_client_error(
                error=exc,
                queue=queue,
                operation="receive_message",
            ) from exc

    @asynchronous
    def _ack_message(
        self,
        *,
        queue: str,
        receipt_handle: str,
    ) -> None:
        queue_url = self._resolve_queue_url(
            queue=queue,
            operation="delete_message",
        )

        try:
            self._sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle,
            )

        except ClientError as exc:
            raise _translate_sqs_client_error(
                error=exc,
                queue=queue,
                operation="delete_message",
            ) from exc

    def _resolve_queue_url(
        self,
        *,
        queue: str,
        operation: str,
    ) -> str:
        if queue.startswith("https://sqs") or queue.startswith("sqs://"):
            return queue

        if url := self._queue_cache.get(queue):
            return url

        try:
            response: Mapping[str, Any] = self._sqs_client.get_queue_url(QueueName=queue)

        except ClientError as exc:
            raise _translate_sqs_client_error(
                error=exc,
                queue=queue,
                operation=operation,
            ) from exc

        queue_url: Any = response.get("QueueUrl")

        if not isinstance(queue_url, str):
            raise AWSError(
                uri=f"sqs://{queue}",
                code="InvalidQueueUrl",
                message=_format_error_message(
                    message="SQS returned an invalid QueueUrl payload",
                    queue=queue,
                    operation=operation,
                ),
            )

        self._queue_cache[queue] = queue_url

        return queue_url


def _format_attribute_value(
    value: RawValue,
) -> dict[str, Any]:
    match value:
        case str() as value:
            return {"DataType": "String", "StringValue": value}

        case int() | float() as value:
            return {"DataType": "Number", "StringValue": str(value)}

        case value:
            raise ValueError(f"Unsupported value {type(value)}")


def _translate_sqs_client_error(
    *,
    error: ClientError,
    queue: str,
    operation: str,
) -> Exception:
    # type: ignore[reportAttributeAccessIssue]
    error_info: Mapping[str, Any] = getattr(error, "response", {}).get("Error", {})
    # type: ignore[reportAttributeAccessIssue]
    response_metadata: Mapping[str, Any] = getattr(error, "response", {}).get(
        "ResponseMetadata",
        {},
    )

    code: str = str(error_info.get("Code") or "").strip()
    message: str = str(error_info.get("Message") or str(error)).strip()
    status_code = response_metadata.get("HTTPStatusCode")
    uri: str = f"sqs://{queue}"

    contextual_message: str = _format_error_message(
        message=message,
        queue=queue,
        operation=operation,
    )
    normalized_code: str = code.lower()

    if (
        normalized_code
        in {
            "awssimplequeueservice.nonexistentqueue",
            "nonexistentqueue",
            "nosuchqueue",
        }
        or status_code == 404  # noqa: PLR2004
    ):
        return AWSResourceNotFound(
            uri=uri,
            code=code or None,
            message=contextual_message,
        )

    if normalized_code in {
        "accessdenied",
        "accessdeniedexception",
        "invalidaccesskeyid",
        "signaturedoesnotmatch",
    } or status_code in {401, 403}:
        return AWSAccessDenied(
            uri=uri,
            code=code or None,
            message=contextual_message,
        )

    return AWSError(
        uri=uri,
        code=code or None,
        message=contextual_message,
    )


def _format_error_message(
    *,
    message: str,
    queue: str,
    operation: str,
) -> str:
    base_message: str = message or "AWS request failed"
    return f"{base_message} [provider=aws_sqs operation={operation} queue={queue}]"
