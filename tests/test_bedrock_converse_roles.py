from itertools import pairwise

import pytest

from draive.bedrock.converse import _context_messages  # pyright: ignore[reportPrivateUsage]
from draive.models import ModelInput, ModelOutput, ModelReasoning
from draive.multimodal import MultimodalContent, TextContent
from draive.resources import ResourceContent


def test_consecutive_inputs_share_a_single_user_message() -> None:
    messages = _context_messages(
        (
            ModelInput.of(MultimodalContent.of("first")),
            ModelInput.of(MultimodalContent.of("second")),
        )
    )

    assert [message["role"] for message in messages] == ["user"]
    assert messages[0]["content"] == [{"text": "first"}, {"text": "second"}]


def test_roles_alternate_when_a_turn_has_no_deliverable_content() -> None:
    messages = _context_messages(
        (
            ModelInput.of(MultimodalContent.of("first")),
            ModelOutput.of(ModelReasoning.of(TextContent.of("thinking"), meta={"kind": "x"})),
            ModelInput.of(MultimodalContent.of("second")),
        )
    )

    roles = [message["role"] for message in messages]
    assert all(left != right for left, right in pairwise(roles))


def test_blank_text_parts_are_not_delivered() -> None:
    messages = _context_messages(
        (
            ModelInput.of(
                MultimodalContent.of(
                    TextContent.of(""),
                    ResourceContent.of(b"png", mime_type="image/png"),
                    TextContent.of("  \n"),
                    TextContent.of("kept"),
                )
            ),
        )
    )

    assert [list(item) for item in messages[0]["content"]] == [["image"], ["text"]]
    assert messages[0]["content"][1] == {"text": "  \nkept"}


def test_turn_with_only_blank_text_is_skipped() -> None:
    messages = _context_messages(
        (
            ModelInput.of(MultimodalContent.of("first")),
            ModelOutput.of(MultimodalContent.of(TextContent.of(""))),
            ModelInput.of(MultimodalContent.of("second")),
        )
    )

    assert [message["role"] for message in messages] == ["user"]


@pytest.mark.parametrize(
    "closing",
    (
        ModelInput.of(MultimodalContent.of(TextContent.of("  "))),
        ModelInput.of(MultimodalContent.empty),
    ),
)
def test_empty_closing_input_after_output_is_rejected(closing: ModelInput) -> None:
    with pytest.raises(ValueError, match="end with a model input"):
        _context_messages(
            (
                ModelInput.of(MultimodalContent.of("first")),
                ModelOutput.of(MultimodalContent.of("answer")),
                closing,
            )
        )


def test_deliberate_assistant_prefill_is_kept() -> None:
    messages = _context_messages(
        (
            ModelInput.of(MultimodalContent.of("first")),
            ModelOutput.of(MultimodalContent.of("prefill")),
        )
    )

    assert [message["role"] for message in messages] == ["user", "assistant"]


def test_empty_closing_input_after_user_turn_is_dropped() -> None:
    messages = _context_messages(
        (
            ModelInput.of(MultimodalContent.of("first")),
            ModelInput.of(MultimodalContent.of(TextContent.of(""))),
        )
    )

    assert [message["role"] for message in messages] == ["user"]
