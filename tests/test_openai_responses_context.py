from base64 import b64encode
from typing import Any

from draive.models import ModelInput, ModelOutput, ModelToolRequest
from draive.multimodal import MultimodalContent
from draive.openai.responses import _context_to_params  # pyright: ignore[reportPrivateUsage]
from draive.resources import ResourceContent

IMAGE: ResourceContent = ResourceContent.of(
    b64encode(b"\x89PNG\r\n\x1a\npretend").decode(),
    mime_type="image/png",
)


def _params(*context: ModelInput | ModelOutput) -> list[dict[str, Any]]:
    return [dict(element) for element in _context_to_params(context, vision_details="auto")]


def test_output_image_is_replayed_as_user_input_image() -> None:
    # `image_generation_call` items are resolved server side through their own id, which
    # never exists for a response requested with `store=False` - a user message carrying
    # the image is the only representation the api accepts for it.
    params = _params(
        ModelInput.of(MultimodalContent.of("draw a circle")),
        ModelOutput.of(MultimodalContent.of("here", IMAGE)),
    )

    assert [param["type"] for param in params] == ["message", "message", "message"]
    assert not any(param["type"] == "image_generation_call" for param in params)

    assistant, image_message = params[1], params[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [{"type": "output_text", "text": "here", "annotations": []}]
    assert image_message["role"] == "user"
    assert image_message["content"] == [
        {
            "type": "input_image",
            "detail": "auto",
            "image_url": IMAGE.to_data_uri(),
        }
    ]


def test_output_image_uses_configured_vision_details() -> None:
    params = [
        dict(element)
        for element in _context_to_params(
            (ModelOutput.of(MultimodalContent.of(IMAGE)),),
            vision_details="low",
        )
    ]
    content: Any = params[0]["content"]

    assert content[0]["detail"] == "low"


def test_consecutive_output_images_share_one_message() -> None:
    params = _params(ModelOutput.of(MultimodalContent.of(IMAGE, IMAGE)))

    assert len(params) == 1
    assert params[0]["role"] == "user"
    assert len(params[0]["content"]) == 2


def test_output_text_around_an_image_keeps_its_order() -> None:
    params = _params(ModelOutput.of(MultimodalContent.of("before", IMAGE, "after")))

    assert [(param["type"], param["role"]) for param in params] == [
        ("message", "assistant"),
        ("message", "user"),
        ("message", "assistant"),
    ]
    assert params[0]["content"][0]["text"] == "before"
    assert params[2]["content"][0]["text"] == "after"


def test_output_image_preceding_a_tool_request_keeps_its_order() -> None:
    params = _params(
        ModelOutput.of(
            MultimodalContent.of(IMAGE),
            ModelToolRequest.of("call_1", tool="draw", arguments={"shape": "square"}),
        )
    )

    assert [param["type"] for param in params] == ["message", "function_call"]
    assert params[0]["role"] == "user"
    assert params[1]["call_id"] == "call_1"
