import json
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Annotated, Any

import pytest
from haiway import Sensitive, State

import draive.fastapi
import draive.starlette


class Profile(State, serializable=True):
    name: str
    visits: int = 0


class Account(State, serializable=True):
    login: str
    secret: Annotated[str, Sensitive()] = "hunter2"


async def _answer(response: Any) -> tuple[int, Mapping[str, str], bytes]:
    started: MutableMapping[str, Any] = {}
    body: list[bytes] = []

    async def receive() -> MutableMapping[str, Any]:
        return {"type": "http.request"}

    async def send(message: MutableMapping[str, Any]) -> None:
        if message["type"] == "http.response.start":
            started.update(message)

        else:
            body.append(message.get("body", b""))

    await response({"type": "http", "method": "GET", "path": "/"}, receive, send)
    headers = {key.decode(): value.decode() for key, value in started["headers"]}
    return started["status"], headers, b"".join(body)


@pytest.mark.parametrize("module", [draive.starlette, draive.fastapi])
def test_state_response_is_exported_by_both_integrations(module: Any) -> None:
    import haiway.starlette

    assert module.StateResponse is haiway.starlette.StateResponse
    assert "StateResponse" in module.__all__


@pytest.mark.asyncio
@pytest.mark.parametrize("module", [draive.starlette, draive.fastapi])
async def test_state_response_answers_with_a_state_as_json(module: Any) -> None:
    status, headers, body = await _answer(module.StateResponse(Profile(name="Ada", visits=3)))

    assert status == 200
    assert headers["content-type"].startswith("application/json")
    assert json.loads(body) == {"name": "Ada", "visits": 3}


@pytest.mark.asyncio
async def test_state_response_answers_with_a_sequence_as_an_array() -> None:
    profiles: Sequence[Profile] = (Profile(name="Ada"), Profile(name="Grace", visits=1))

    status, _, body = await _answer(draive.starlette.StateResponse(profiles, status_code=201))

    assert status == 201
    assert json.loads(body) == [{"name": "Ada", "visits": 0}, {"name": "Grace", "visits": 1}]


@pytest.mark.asyncio
async def test_state_response_body_carries_sensitive_values() -> None:
    # `Sensitive` redacts representations and logs, not what is written to a body
    _, _, body = await _answer(draive.starlette.StateResponse(Account(login="ada")))

    assert json.loads(body) == {"login": "ada", "secret": "hunter2"}
