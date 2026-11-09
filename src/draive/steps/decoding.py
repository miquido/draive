import json
from collections.abc import Callable

from haiway import State, ctx

from draive.multimodal import MultimodalContent

__all__ = (
    "decode_content",
    "json_payload",
)


def json_payload(
    content: MultimodalContent,
    /,
) -> str:
    unwrapped: str = without_code_fence(content.to_str().strip())
    try:
        # models occasionally continue past the requested object, appending prose
        # or even a second object - only the first complete value is the result
        _, payload_end = json.JSONDecoder().raw_decode(unwrapped)

    except ValueError:
        return unwrapped  # leave decoding errors to the caller

    return unwrapped[:payload_end]


def without_code_fence(
    content: str,
    /,
) -> str:
    # models without a native json output mode tend to wrap the payload
    # within a markdown code fence despite being asked not to
    if not content.startswith("```"):
        return content

    opening_end: int = content.find("\n")
    if opening_end < 0:
        return content

    body: str = content[opening_end + 1 :]
    closing_start: int = body.rfind("```")

    return body[:closing_start].strip() if closing_start >= 0 else body.strip()


def decode_content[Decoded: State](
    content: MultimodalContent,
    decoded: type[Decoded],
    /,
    *,
    decoder: Callable[[MultimodalContent], Decoded] | None = None,
) -> Decoded:
    try:
        if decoder is not None:
            ctx.log_debug("...decoding result...")
            return decoder(content)

        elif artifacts := content.artifacts(category="json"):
            ctx.log_debug("...direct artifact found!")
            return artifacts[-1].to_state(decoded)

        else:  # fallback to default decoding
            ctx.log_debug("...decoding result...")
            return decoded.from_json(json_payload(content))

    except Exception as exc:
        ctx.log_error(
            f"Failed to decode {decoded.__name__} model due to an error: {type(exc)}",
            exception=exc,
        )
        raise exc
