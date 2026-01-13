from draive.anthropic.messages import _context_messages
from draive.models import ModelOutput, ModelReasoning
from draive.multimodal import MultimodalContent


def test_context_messages_preserves_redacted_thinking_data_in_meta() -> None:
    messages = list(
        _context_messages(
            (
                ModelOutput.of(
                    ModelReasoning.of(
                        MultimodalContent.empty,
                        meta={
                            "kind": "redacted_thinking",
                            "data": "opaque-redacted-data",
                        },
                    )
                ),
            ),
            prefill=None,
            output="text",
        )
    )

    assert messages == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "redacted_thinking",
                    "data": "opaque-redacted-data",
                }
            ],
        }
    ]
