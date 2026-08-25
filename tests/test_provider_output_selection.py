"""Unsupported output modalities have to fail instead of silently answering with text."""

from typing import Any

import pytest
from haiway import MISSING, State
from openai import Omit

from draive.anthropic.messages import _output_config as _anthropic_output_config
from draive.bedrock.converse import _verify_output as _bedrock_verify_output
from draive.gemini.generating import _output_config as _gemini_output_config
from draive.mistral.completions import _response_format as _mistral_response_format
from draive.models import ModelOutputSelection
from draive.ollama.chat import _response_format as _ollama_response_format
from draive.openai.responses import _text_output
from draive.vllm.messages import _response_format as _vllm_response_format

# accepted everywhere - a text answer matches what was asked for
SUPPORTED: tuple[ModelOutputSelection, ...] = (
    "auto",
    "text",
    "json",
    ("text",),
    ("text", "image"),
)
# rejected everywhere - no provider produces video
UNSUPPORTED: tuple[ModelOutputSelection, ...] = (
    "video",
    ("video",),
    ("audio", "video"),
)
# accepted only by providers able to return media of their own
MEDIA_ONLY: tuple[ModelOutputSelection, ...] = (
    "image",
    "audio",
    ("image",),
    ("audio",),
)


def _anthropic(output: ModelOutputSelection) -> Any:
    return _anthropic_output_config(output, effort=MISSING)


# every provider entry point resolving a requested output selection
CONVERTERS = (
    _text_output,
    _anthropic,
    _gemini_output_config,
    _mistral_response_format,
    _ollama_response_format,
    _vllm_response_format,
    _bedrock_verify_output,
)
# providers returning modalities beyond text
MEDIA_CAPABLE = (_gemini_output_config,)


@pytest.mark.parametrize("convert", CONVERTERS)
@pytest.mark.parametrize("output", SUPPORTED)
def test_supported_output_selection_is_accepted(
    convert: Any,
    output: ModelOutputSelection,
) -> None:
    convert(output)


@pytest.mark.parametrize("convert", CONVERTERS)
@pytest.mark.parametrize("output", UNSUPPORTED)
def test_unsupported_output_selection_is_rejected(
    convert: Any,
    output: ModelOutputSelection,
) -> None:
    with pytest.raises(NotImplementedError):
        convert(output)


@pytest.mark.parametrize("convert", MEDIA_CAPABLE)
@pytest.mark.parametrize("output", MEDIA_ONLY)
def test_media_output_selection_is_accepted_by_capable_providers(
    convert: Any,
    output: ModelOutputSelection,
) -> None:
    convert(output)


@pytest.mark.parametrize("convert", tuple(c for c in CONVERTERS if c not in MEDIA_CAPABLE))
@pytest.mark.parametrize("output", MEDIA_ONLY)
def test_media_output_selection_is_rejected_by_text_only_providers(
    convert: Any,
    output: ModelOutputSelection,
) -> None:
    with pytest.raises(NotImplementedError):
        convert(output)


class TestOpenAIJsonOutput:
    def test_a_schema_selection_requests_the_structured_format(self) -> None:
        class Example(State, serializable=True):
            name: str

        config = _text_output(Example)

        assert not isinstance(config, Omit)
        assert config["format"]["type"] == "json_schema"

    def test_a_schema_less_selection_requests_no_format(self) -> None:
        # json_object requires the word "json" within the input itself, which
        # cannot be satisfied without synthesizing content into the request
        assert isinstance(_text_output("json"), Omit)

    def test_a_schema_less_selection_still_carries_verbosity(self) -> None:
        config = _text_output("json", verbosity="low")

        assert not isinstance(config, Omit)
        assert config == {"verbosity": "low"}
