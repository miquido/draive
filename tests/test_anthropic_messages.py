import pytest
from anthropic import omit
from haiway import MISSING, State, as_dict, ctx

from draive.anthropic.config import AnthropicConfig
from draive.anthropic.messages import (
    AnthropicMessages,
    _context_messages,
    _output_config,
    _thinking_config,
    _tools_as_tool_params,
)
from draive.models import (
    ModelInput,
    ModelInputInvalid,
    ModelOutput,
    ModelReasoning,
    ModelTools,
    ModelToolSpecification,
)
from draive.multimodal import MultimodalContent


def test_context_messages_preserves_redacted_thinking_data_in_meta() -> None:
    messages = list(
        _context_messages(
            (
                ModelInput.of(MultimodalContent.of("Describe Paris.")),
                ModelOutput.of(
                    ModelReasoning.of(
                        MultimodalContent.empty,
                        meta={
                            "kind": "redacted_thinking",
                            "data": "opaque-redacted-data",
                        },
                    )
                ),
                ModelInput.of(MultimodalContent.of("And Kraków?")),
            ),
        )
    )

    assert messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "redacted_thinking",
                "data": "opaque-redacted-data",
            }
        ],
    }


def test_context_messages_carry_the_context_verbatim() -> None:
    messages = list(
        _context_messages(
            (ModelInput.of(MultimodalContent.of("Describe Paris.")),),
        )
    )

    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Describe Paris."}],
        }
    ]


def test_context_messages_rejects_trailing_model_turn() -> None:
    # assistant prefill is unsupported, a trailing model turn would become one
    with pytest.raises(ValueError) as error:
        list(
            _context_messages(
                (
                    ModelInput.of(MultimodalContent.of("Describe Paris.")),
                    ModelOutput.of(MultimodalContent.of("Paris is")),
                ),
            )
        )

    # the rejection has to name the prefill restriction it stands for - it surfaces
    # as the `reason` of the `ModelInputInvalid` raised to the caller
    assert "prefill" in str(error.value)


def test_output_config_carries_schema_for_state_output() -> None:
    class Example(State, serializable=True):
        name: str

    assert _output_config(Example, effort=MISSING) == {
        "format": {
            "type": "json_schema",
            "schema": as_dict(Example.__SPECIFICATION__),
        },
    }


def test_output_config_omitted_for_non_schema_output() -> None:
    assert _output_config("json", effort=MISSING) is omit
    assert _output_config("text", effort=MISSING) is omit
    assert _output_config("auto", effort=MISSING) is omit


def test_output_config_carries_effort_level() -> None:
    assert _output_config("text", effort="high") == {"effort": "high"}


def test_thinking_config_maps_requested_mode() -> None:
    assert _thinking_config("adaptive") == {"type": "adaptive"}
    assert _thinking_config("disabled") == {"type": "disabled"}


def test_thinking_config_omitted_without_configuration() -> None:
    # not every model accepts an explicit mode, the api default is adaptive anyway
    assert _thinking_config(MISSING) is omit


def test_tools_none_selection_omits_declarations() -> None:
    tool_choice, tools = _tools_as_tool_params(
        selection="none",
        specification=(ModelToolSpecification.of(name="ping"),),
    )

    assert tool_choice is omit
    assert tools is omit


@pytest.mark.asyncio
async def test_invalid_context_reports_the_rejection_reason() -> None:
    """Regression test: wrapping the context conversion failure without its message
    left `ModelInputInvalid` saying only "Invalid input", with the explanation
    reachable solely through `__cause__` - and its declared `reason` slot unset.
    """
    model = object.__new__(AnthropicMessages)
    model._provider = "anthropic"
    model._client = None  # never reached, the context is rejected first

    async with ctx.scope("test"):
        stream = model.completion(
            instructions="system",
            tools=ModelTools.none,
            context=(
                ModelInput.of(MultimodalContent.of("Describe Paris.")),
                ModelOutput.of(MultimodalContent.of("Paris is")),
            ),
            output="text",
            config=AnthropicConfig(model="claude-opus-5"),
        )
        with pytest.raises(ModelInputInvalid) as error:
            async for _ in stream:
                pass

    assert error.value.reason is not None
    assert "prefill" in error.value.reason
    assert "prefill" in str(error.value)
