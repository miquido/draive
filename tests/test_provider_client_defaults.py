from collections.abc import Callable
from typing import Any

import pytest
from openai import OpenAIError

from draive.anthropic.api import AnthropicAPI
from draive.openai.api import OpenAIAPI
from draive.vllm.api import VLLMAPI

# both SDKs apply their own default only when the argument is omitted -
# passing `None` explicitly means "no timeout at all"
_SDK_DEFAULT_READ_TIMEOUT = 600

type _ClientFactory = Callable[[], Any]


def _openai(timeout: float | None = None) -> _ClientFactory:
    return lambda: OpenAIAPI(api_key="k", timeout=timeout)._prepare_client()


def _vllm(timeout: float | None = None) -> _ClientFactory:
    return lambda: VLLMAPI(base_url="http://vllm/v1", timeout=timeout)._prepare_client()


def _anthropic(timeout: float | None = None) -> _ClientFactory:
    return lambda: AnthropicAPI(api_key="k", timeout=timeout)._prepare_client()


@pytest.mark.parametrize(
    "prepare",
    [
        pytest.param(_openai(), id="openai"),
        pytest.param(_vllm(), id="vllm"),
        pytest.param(_anthropic(), id="anthropic"),
    ],
)
def test_client_without_timeout_keeps_the_sdk_default(prepare: _ClientFactory) -> None:
    timeout: Any = prepare().timeout

    assert timeout is not None, "an unset timeout must not disable timeouts entirely"
    assert timeout.read == _SDK_DEFAULT_READ_TIMEOUT


@pytest.mark.parametrize(
    "prepare",
    [
        pytest.param(_openai(12.0), id="openai"),
        pytest.param(_vllm(12.0), id="vllm"),
        pytest.param(_anthropic(12.0), id="anthropic"),
    ],
)
def test_explicit_timeout_is_honored(prepare: _ClientFactory) -> None:
    assert prepare().timeout == 12.0


def test_vllm_never_sends_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # an actual OpenAI credential must never reach an arbitrary vLLM base_url
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")

    assert _vllm()().api_key == "vllm"


def test_vllm_constructs_without_any_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # vLLM endpoints need no credentials, construction must not require one
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    assert _vllm()() is not None


def test_vllm_forwards_an_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # the placeholder is only a default - an endpoint started with `--api-key` needs its own
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")

    client = VLLMAPI(
        base_url="http://vllm/v1",
        api_key="vllm-endpoint-secret",
        organization="acme",
    )._prepare_client()

    assert client.api_key == "vllm-endpoint-secret"
    assert client.organization == "acme"
    # untouched placeholders still stand in
    assert client.project == "vllm"


def test_vllm_extra_arguments_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = VLLMAPI(base_url="http://vllm/v1", max_retries=7)._prepare_client()

    assert client.max_retries == 7
    assert client.api_key == "vllm"


def test_vllm_reads_the_endpoint_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # a dedicated env var allows configuring the endpoint credential without leaking OPENAI_API_KEY
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-endpoint-secret")

    assert _vllm()().api_key == "vllm-endpoint-secret"


def test_vllm_api_key_does_not_collide_with_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # `api_key` is an explicit argument - it can't be duplicated through extra arguments
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    client = VLLMAPI(
        **{
            "base_url": "http://vllm/v1",
            "api_key": "vllm-endpoint-secret",
        }
    )._prepare_client()

    assert client.api_key == "vllm-endpoint-secret"


def test_openai_constructs_for_a_custom_base_url_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenAI compatible servers may require no credentials at all
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    client = OpenAIAPI(base_url="http://localhost:1234/v1")._prepare_client()

    assert client.api_key  # the SDK rejects a falsy key


def test_openai_requires_credentials_without_a_custom_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # without a custom base_url the request would reach actual OpenAI - let the SDK complain
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    with pytest.raises(OpenAIError):
        OpenAIAPI()._prepare_client()


def test_openai_azure_falls_back_to_the_ad_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # an empty key would slip past the Azure credentials check and fail within the base client
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "ad-token")

    client = OpenAIAPI(
        azure_api_endpoint="https://example.openai.azure.com",
        azure_api_version="2024-10-21",
    )._prepare_client()

    assert client._azure_ad_token == "ad-token"


def test_openai_prefers_the_azure_key_over_the_openai_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert OpenAIAPI()._prepare_client().api_key == "azure-key"
