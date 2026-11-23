import pytest

from draive.gemini.config import GeminiConfig
from draive.gemini.utils import thinking_config


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (0, None),
        (-1, {"include_thoughts": True, "thinking_budget": -1}),
        (1024, {"include_thoughts": True, "thinking_budget": 1024}),
    ],
)
def test_thinking_budget_zero_skips_thinking_config(
    budget: int,
    expected: dict[str, object] | None,
) -> None:
    assert thinking_config(GeminiConfig(model="gemini-test", thinking_budget=budget)) == expected


def test_thinking_config_omitted_without_configuration() -> None:
    assert thinking_config(GeminiConfig(model="gemini-test")) is None
