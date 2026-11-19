from typing import Any

import pytest
from google.genai.types import HttpOptionsDict

from draive.gemini import api as gemini_api
from draive.gemini.api import GeminiAPI


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vertexai", "http_options", "environ", "expected"),
    [
        (None, None, {}, {"api_version": "v1beta"}),
        (None, {"timeout": 10}, {}, {"timeout": 10, "api_version": "v1beta"}),
        (None, {"api_version": "v1"}, {}, {"api_version": "v1"}),
        (True, None, {}, None),
        (None, None, {"GOOGLE_GENAI_USE_VERTEXAI": "true"}, None),
        (None, None, {"GOOGLE_GENAI_USE_ENTERPRISE": "true"}, None),
        (None, None, {"GOOGLE_GENAI_API_VERSION": "v1"}, None),
    ],
)
async def test_client_prefers_v1beta_for_the_developer_api(
    vertexai: bool | None,
    http_options: HttpOptionsDict | None,
    environ: dict[str, str],
    expected: dict[str, Any] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_GENAI_API_VERSION",
    ):
        monkeypatch.delenv(key, raising=False)

    for key, value in environ.items():
        monkeypatch.setenv(key, value)

    captured: dict[str, Any] = {}

    def client(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(gemini_api, "Client", client)
    await GeminiAPI(
        api_key="test", vertexai=vertexai, http_options=http_options
    )._initialize_client()

    assert captured["http_options"] == expected
    # caller provided options are never mutated
    if http_options is not None:
        assert "api_version" not in http_options or http_options["api_version"] == "v1"
