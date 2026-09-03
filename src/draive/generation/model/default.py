import json
from collections.abc import Iterable
from typing import Any

from haiway import State, ctx

from draive.generation.model.types import ModelGenerationDecoder
from draive.models import (
    ModelInput,
    ModelInstructions,
    ModelOutput,
)
from draive.multimodal import ArtifactContent, Multimodal, MultimodalContent
from draive.steps import Step
from draive.tools import Toolbox

__all__ = ("generate_model",)


def _json_payload(
    content: MultimodalContent,
    /,
) -> str:
    unwrapped: str = _without_code_fence(content.to_str().strip())
    try:
        # models occasionally continue past the requested object, appending prose
        # or even a second object - only the first complete value is the result
        _, payload_end = json.JSONDecoder().raw_decode(unwrapped)

    except ValueError:
        return unwrapped  # leave decoding errors to the caller

    return unwrapped[:payload_end]


def _without_code_fence(
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


async def generate_model[Generated: State](
    generated: type[Generated],
    /,
    *,
    instructions: ModelInstructions,
    input: Multimodal,  # noqa: A002
    toolbox: Toolbox,
    examples: Iterable[tuple[Multimodal, Generated]],
    decoder: ModelGenerationDecoder[Generated] | None,
    **extra: Any,
) -> Generated:
    completion: MultimodalContent = await Step.looping_completion(
        instructions=instructions,
        tools=toolbox,
        output=generated,
        **extra,
    ).run(
        (
            *(
                message
                for example in examples
                for message in (
                    ModelInput.of(MultimodalContent.of(example[0])),
                    ModelOutput.of(MultimodalContent.of(ArtifactContent.of(example[1]))),
                )
            ),
            ModelInput.of(MultimodalContent.of(input)),
        )
    )

    try:
        if decoder is not None:
            ctx.log_debug("...decoding result...")
            return decoder(completion)

        elif artifacts := completion.artifacts(category="json"):
            ctx.log_debug("...direct artifact found!")
            return artifacts[0].to_state(generated)

        else:  # fallback to default decoding
            ctx.log_debug("...decoding result...")
            return generated.from_json(_json_payload(completion))

    except Exception as exc:
        ctx.log_error(
            f"Failed to decode {generated.__name__} model due to an error: {type(exc)}",
            exception=exc,
        )
        raise exc
