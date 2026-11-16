from typing import Any, ClassVar

import pytest
from haiway import ctx

from draive.aws import api as aws_api
from draive.aws.api import AWSAPI


class _FakeSession:
    instances: ClassVar[list[_FakeSession]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.region_name = kwargs.get("region_name")
        self.created: list[str] = []
        _FakeSession.instances.append(self)

    def client(self, service_name: str) -> _FakeClient:
        self.created.append(service_name)
        return _FakeClient(service_name)


class _FakeClient:
    def __init__(self, service: str) -> None:
        self.service = service
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def fake_session(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeSession.instances = []
    monkeypatch.setattr(aws_api, "Session", _FakeSession)


def test_session_receives_only_provided_arguments() -> None:
    AWSAPI()
    AWSAPI(
        region_name="eu-west-1",
        access_key_id="key",
        secret_access_key="secret",  # nosec: B106
        profile_name="profile",
    )

    assert _FakeSession.instances[0].kwargs == {}
    assert _FakeSession.instances[1].kwargs == {
        "aws_access_key_id": "key",
        "aws_secret_access_key": "secret",
        "region_name": "eu-west-1",
        "profile_name": "profile",
    }


def test_region_reflects_session() -> None:
    assert AWSAPI(region_name="us-east-1").region == "us-east-1"


@pytest.mark.asyncio
async def test_prepare_clients_creates_expected_services() -> None:
    api = AWSAPI()

    async with ctx.scope("test"):
        await api._prepare_s3_client()  # pyright: ignore[reportPrivateUsage]
        await api._prepare_sqs_client()  # pyright: ignore[reportPrivateUsage]
        await api._prepare_cloudwatch_clients()  # pyright: ignore[reportPrivateUsage]

    assert _FakeSession.instances[0].created == ["s3", "sqs", "cloudwatch", "logs", "events"]


@pytest.mark.asyncio
async def test_dispose_closes_every_prepared_client_once() -> None:
    api = AWSAPI()

    async with ctx.scope("test"):
        await api._prepare_s3_client()  # pyright: ignore[reportPrivateUsage]
        await api._prepare_cloudwatch_clients()  # pyright: ignore[reportPrivateUsage]
        clients: list[_FakeClient] = [
            api._s3_client,  # pyright: ignore[reportPrivateUsage]
            api._cloudwatch_client,  # pyright: ignore[reportPrivateUsage]
            api._cloudwatch_logs_client,  # pyright: ignore[reportPrivateUsage]
            api._eventbridge_client,  # pyright: ignore[reportPrivateUsage]
        ]
        await api._dispose_clients()  # pyright: ignore[reportPrivateUsage]
        await api._dispose_clients()  # pyright: ignore[reportPrivateUsage]

    assert [client.closed for client in clients] == [1, 1, 1, 1]
