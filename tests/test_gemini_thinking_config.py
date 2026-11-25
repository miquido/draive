from typing import Literal

import pytest
from google.genai.types import ThinkingLevel

from draive.gemini.config import GeminiConfig
from draive.gemini.utils import thinking_config


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("minimal", ThinkingLevel.MINIMAL),
        ("low", ThinkingLevel.LOW),
        ("medium", ThinkingLevel.MEDIUM),
        ("high", ThinkingLevel.HIGH),
        ("xhigh", ThinkingLevel.XHIGH),
    ],
)
def test_thinking_level_maps_to_the_sdk_level(
    level: Literal["minimal", "low", "medium", "high", "xhigh"],
    expected: ThinkingLevel,
) -> None:
    assert thinking_config(GeminiConfig(model="gemini-test", thinking_level=level)) == {
        "include_thoughts": True,
        "thinking_level": expected,
    }


def test_thinking_config_omitted_without_configuration() -> None:
    assert thinking_config(GeminiConfig(model="gemini-test")) is None


def test_thinking_budget_is_not_configurable() -> None:
    assert not hasattr(GeminiConfig(model="gemini-test"), "thinking_budget")
