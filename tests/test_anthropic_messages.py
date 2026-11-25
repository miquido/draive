import pytest
from anthropic import omit
from haiway import Missing, State, as_dict, ctx

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

    assert _output_config(Example, effort=Missing) == {
        "format": {
            "type": "json_schema",
            "schema": as_dict(Example.__SPECIFICATION__),
        },
    }


def test_output_config_omitted_for_non_schema_output() -> None:
    assert _output_config("json", effort=Missing) is omit
    assert _output_config("text", effort=Missing) is omit
    assert _output_config("auto", effort=Missing) is omit


def test_output_config_carries_effort_level() -> None:
    assert _output_config("text", effort="high") == {"effort": "high"}


def test_thinking_config_maps_requested_mode() -> None:
    assert _thinking_config("adaptive") == {"type": "adaptive"}
    assert _thinking_config("disabled") == {"type": "disabled"}
    assert _thinking_config("between_tools") == {"type": "between_tools"}


def test_thinking_config_omitted_without_configuration() -> None:
    # not every model accepts an explicit mode, the api default is adaptive anyway
    assert _thinking_config(Missing) is omit


def test_tools_none_selection_omits_declarations() -> None:
    tool_choice, tools = _tools_as_tool_params(
        selection="none",
        specification=(ModelToolSpecification.of(name="ping"),),
    )

    assert tool_choice is omit
    assert tools is omit


@pytest.mark.asyncio
async def test_tools_required_selection_falls_back_to_auto() -> None:
    async with ctx.scope("test"):
        tool_choice, tools = _tools_as_tool_params(
            selection="required",
            specification=(
                ModelToolSpecification.of(name="ping"),
                ModelToolSpecification.of(name="pong"),
            ),
        )

    assert tool_choice == {"type": "auto"}
    assert tools is not omit
    assert [tool["name"] for tool in tools] == ["ping", "pong"]


@pytest.mark.asyncio
async def test_tools_specific_selection_falls_back_to_auto() -> None:
    pong = ModelToolSpecification.of(name="pong")
    async with ctx.scope("test"):
        tool_choice, tools = _tools_as_tool_params(
            selection=pong,
            specification=(ModelToolSpecification.of(name="ping"), pong),
        )

    assert tool_choice == {"type": "auto"}
    assert tools is not omit
    assert [tool["name"] for tool in tools] == ["ping", "pong"]


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
            config=AnthropicConfig(model="claude-opus-5-5"),
        )
        with pytest.raises(ModelInputInvalid) as error:
            async for _ in stream:
                pass

    assert error.value.reason is not None
    assert "prefill" in error.value.reason
    assert "prefill" in str(error.value)


def test_context_messages_skip_blank_text_blocks() -> None:
    from draive.multimodal import TextContent
    from draive.resources import ResourceContent

    messages = list(
        _context_messages(
            (
                ModelInput.of(
                    MultimodalContent.of(
                        TextContent.of(""),
                        ResourceContent.of(b"png", mime_type="image/png"),
                        TextContent.of(" \n"),
                    )
                ),
            ),
        )
    )

    assert [block["type"] for block in messages[0]["content"]] == ["image"]


def test_context_messages_skip_turns_without_deliverable_content() -> None:
    from draive.multimodal import TextContent

    messages = list(
        _context_messages(
            (
                ModelInput.of(MultimodalContent.of("first")),
                ModelOutput.of(MultimodalContent.of(TextContent.of(""))),
                ModelInput.of(MultimodalContent.of("second")),
            ),
        )
    )

    assert [message["role"] for message in messages] == ["user", "user"]


def test_context_messages_reject_context_ending_without_deliverable_input() -> None:
    from draive.multimodal import TextContent

    with pytest.raises(ValueError, match="model input"):
        list(
            _context_messages(
                (
                    ModelInput.of(MultimodalContent.of("first")),
                    ModelOutput.of(MultimodalContent.of("answer")),
                    ModelInput.of(MultimodalContent.of(TextContent.of(""))),
                ),
            )
        )


def test_foreign_reasoning_is_dropped() -> None:
    messages = list(
        _context_messages(
            (
                ModelInput.of(MultimodalContent.of("first")),
                ModelOutput.of(
                    ModelReasoning.of(MultimodalContent.of("thought"), meta={"kind": "thought"}),
                    MultimodalContent.of("answer"),
                ),
                ModelInput.of(MultimodalContent.of("second")),
            ),
        )
    )

    assert messages[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "answer"}],
    }
