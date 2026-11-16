from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
from botocore.exceptions import ClientError
from haiway import MQMessage, MQQueue, ctx

from draive.aws.sqs import AWSSQSMixin
from draive.aws.types import AWSAccessDenied, AWSError, AWSResourceNotFound


def _client_error(code: str, status: int = 400) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "failure"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "operation",
    )


class _FakeSQSClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []
        self.visibility_changes: list[dict[str, Any]] = []
        self.queue_lookups: list[str] = []
        self.receive_calls: int = 0
        self.batches: list[list[dict[str, Any]]] = []
        self.failure: ClientError | None = None
        self.closed: bool = False

    def get_queue_url(self, QueueName: str) -> Mapping[str, Any]:
        self.queue_lookups.append(QueueName)
        if self.failure is not None:
            raise self.failure

        return {"QueueUrl": f"https://sqs.eu-west-1.amazonaws.com/123/{QueueName}"}

    def send_message(self, **kwargs: Any) -> None:
        if self.failure is not None:
            raise self.failure

        self.sent.append(kwargs)

    def receive_message(self, **kwargs: Any) -> Mapping[str, Any]:
        self.receive_calls += 1
        if self.failure is not None:
            raise self.failure

        if self.batches:
            return {"Messages": self.batches.pop(0)}

        return {}

    def delete_message(self, **kwargs: Any) -> None:
        if self.failure is not None:
            raise self.failure

        self.deleted.append(kwargs)

    def change_message_visibility(self, **kwargs: Any) -> None:
        if self.failure is not None:
            raise self.failure

        self.visibility_changes.append(kwargs)

    def close(self) -> None:
        self.closed = True


def _mixin(client: _FakeSQSClient) -> AWSSQSMixin:
    mixin = AWSSQSMixin(region_name="eu-west-1")
    mixin._sqs_client = client  # pyright: ignore[reportPrivateUsage]
    return mixin


async def _queue(
    mixin: AWSSQSMixin,
    name: str = "jobs",
) -> AbstractAsyncContextManager[MQQueue[str]]:
    return await mixin._queue_access(  # pyright: ignore[reportPrivateUsage]
        name,
        content_encoder=str,
        content_decoder=str,
    )


@pytest.mark.asyncio
async def test_publish_encodes_content_and_attributes() -> None:
    client = _FakeSQSClient()
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            assert isinstance(queue, MQQueue)
            await queue.publish(
                "hello",
                attributes={"name": "value", "count": 3, "ratio": 0.5},
            )

    assert client.sent == [
        {
            "QueueUrl": "https://sqs.eu-west-1.amazonaws.com/123/jobs",
            "MessageBody": "hello",
            "MessageAttributes": {
                "name": {"DataType": "String", "StringValue": "value"},
                "count": {"DataType": "Number", "StringValue": "3"},
                "ratio": {"DataType": "Number", "StringValue": "0.5"},
            },
        }
    ]


@pytest.mark.asyncio
async def test_publish_without_attributes_sends_empty_mapping() -> None:
    client = _FakeSQSClient()
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            await queue.publish("hello")

    assert client.sent[0]["MessageAttributes"] == {}


@pytest.mark.asyncio
async def test_publish_formats_boolean_attribute_as_valid_sqs_value() -> None:
    client = _FakeSQSClient()
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            await queue.publish("hello", attributes={"flag": True})

    value: Mapping[str, str] = client.sent[0]["MessageAttributes"]["flag"]
    if value["DataType"] == "Number":
        float(value["StringValue"])  # must parse as a number

    else:
        assert value["StringValue"] == "true"


@pytest.mark.asyncio
async def test_publish_rejects_unsupported_attribute_value() -> None:
    client = _FakeSQSClient()
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            with pytest.raises(ValueError, match="Unsupported value"):
                await queue.publish("hello", attributes={"missing": None})

    assert client.sent == []


@pytest.mark.asyncio
async def test_queue_url_is_resolved_once_and_cached() -> None:
    client = _FakeSQSClient()
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin, "sqs://jobs") as queue:
            await queue.publish("a")
            await queue.publish("b")

    assert client.queue_lookups == ["jobs"]
    assert len(client.sent) == 2


@pytest.mark.asyncio
async def test_queue_url_is_used_directly() -> None:
    client = _FakeSQSClient()
    mixin = _mixin(client)
    url = "https://sqs.eu-west-1.amazonaws.com/123/direct"

    async with ctx.scope("test"):
        async with await _queue(mixin, url) as queue:
            await queue.publish("a")

    assert client.queue_lookups == []
    assert client.sent[0]["QueueUrl"] == url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "status", "expected"),
    (
        pytest.param(
            "AWS.SimpleQueueService.NonExistentQueue",
            400,
            AWSResourceNotFound,
        ),
        ("NoSuchQueue", 400, AWSResourceNotFound),
        ("Anything", 404, AWSResourceNotFound),
        ("AccessDenied", 403, AWSAccessDenied),
        ("InvalidAccessKeyId", 403, AWSAccessDenied),
        ("Anything", 401, AWSAccessDenied),
        ("Throttling", 400, AWSError),
    ),
)
async def test_errors_are_translated(
    code: str,
    status: int,
    expected: type[Exception],
) -> None:
    client = _FakeSQSClient()
    mixin = _mixin(client)
    client.failure = _client_error(code, status)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            with pytest.raises(expected) as raised:
                await queue.publish("a")

    assert type(raised.value) is expected
    assert "provider=aws_sqs" in str(raised.value)
    assert "queue=jobs" in str(raised.value)


@pytest.mark.asyncio
async def test_invalid_queue_url_payload_is_reported() -> None:
    client = _FakeSQSClient()
    client.get_queue_url = lambda QueueName: {}
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            with pytest.raises(AWSError, match="invalid QueueUrl"):
                await queue.publish("a")


@pytest.mark.asyncio
async def test_consume_yields_messages_and_acknowledges_by_receipt() -> None:
    client = _FakeSQSClient()
    client.batches = [
        [{"ReceiptHandle": "r1", "Body": "one"}],
        [{"ReceiptHandle": "r2", "Body": "two"}],
    ]
    mixin = _mixin(client)
    received: list[str] = []

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            async with await queue.consume() as messages:
                async for message in messages:
                    assert isinstance(message, MQMessage)
                    received.append(message.content)
                    await message.acknowledge()
                    if len(received) == 2:
                        break

    assert received == ["one", "two"]
    assert [call["ReceiptHandle"] for call in client.deleted] == ["r1", "r2"]
    assert all(call["QueueUrl"].endswith("/jobs") for call in client.deleted)


@pytest.mark.asyncio
async def test_consume_stops_polling_after_leaving_context() -> None:
    client = _FakeSQSClient()
    client.batches = [[{"ReceiptHandle": "r1", "Body": "one"}]]
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            async with await queue.consume() as messages:
                async for _ in messages:
                    break

            calls: int = client.receive_calls

    assert client.receive_calls == calls == 1


@pytest.mark.asyncio
async def test_reject_does_not_delete_message() -> None:
    client = _FakeSQSClient()
    client.batches = [[{"ReceiptHandle": "r1", "Body": "one"}]]
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            async with await queue.consume() as messages:
                async for message in messages:
                    await message.reject()
                    break

    assert client.deleted == []
    # rejected messages become visible again immediately instead of waiting out the timeout
    assert client.visibility_changes == [
        {
            "QueueUrl": "https://sqs.eu-west-1.amazonaws.com/123/jobs",
            "ReceiptHandle": "r1",
            "VisibilityTimeout": 0,
        }
    ]


@pytest.mark.asyncio
async def test_consume_receive_error_is_translated() -> None:
    client = _FakeSQSClient()
    client.failure = _client_error("AccessDenied", 403)
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            async with await queue.consume() as messages:
                with pytest.raises(AWSAccessDenied):
                    async for _ in messages:
                        pass


@pytest.mark.asyncio
async def test_dispose_closes_clients_and_clears_queue_cache() -> None:
    client = _FakeSQSClient()
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin) as queue:
            await queue.publish("a")

        assert mixin._queue_cache  # pyright: ignore[reportPrivateUsage]
        await mixin._dispose_clients()  # pyright: ignore[reportPrivateUsage]

    assert client.closed
    assert not mixin._queue_cache  # pyright: ignore[reportPrivateUsage]
    assert not hasattr(mixin, "_sqs_client")


@pytest.mark.asyncio
async def test_dispose_continues_after_client_close_failure() -> None:
    first = _FakeSQSClient()
    second = _FakeSQSClient()

    def failing_close() -> None:
        raise RuntimeError("close failed")

    first.close = failing_close
    mixin = AWSSQSMixin(region_name="eu-west-1")
    mixin._s3_client = first  # pyright: ignore[reportPrivateUsage]
    mixin._sqs_client = second  # pyright: ignore[reportPrivateUsage]

    async with ctx.scope("test"):
        await mixin._dispose_clients()  # pyright: ignore[reportPrivateUsage]

    assert second.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "http://localhost:4566/000000000000/jobs",
        "http://sqs.localhost.localstack.cloud:4566/000000000000/jobs",
        "https://localhost.localstack.cloud:4566/000000000000/jobs",
        "https://sqs.eu-west-1.amazonaws.com/123/direct",
    ),
)
async def test_http_queue_urls_are_used_directly(url: str) -> None:
    client = _FakeSQSClient()
    mixin = _mixin(client)

    async with ctx.scope("test"):
        async with await _queue(mixin, url) as queue:
            await queue.publish("a")

    assert client.queue_lookups == []
    assert client.sent[0]["QueueUrl"] == url
