from pathlib import Path
from typing import Any, cast

from sentencepiece import SentencePieceProcessor  # pyright: ignore[reportMissingTypeStubs]

from draive.mistral.config import MistralChatConfig
from draive.scope import ctx
from draive.utils import Missing, cache, not_missing

__all__ = [
    "mistral_tokenize_text",
]


def mistral_tokenize_text(
    text: str,
    **extra: Any,
) -> list[int]:
    model_name: str | Missing = ctx.state(MistralChatConfig).model
    if not_missing(model_name):
        return cast(
            list[int],
            _encoding(model_name=model_name).encode(text),  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        )

    else:
        raise ValueError("Missing model name in MistralChatConfig")


@cache(limit=4)
def _encoding(model_name: str) -> SentencePieceProcessor:
    if model_file := _mapping.get(model_name):
        return SentencePieceProcessor(model_file=str(Path(__file__).parent / "tokens" / model_file))  # pyright: ignore[reportCallIssue]

    else:
        raise NotImplementedError("Requested unknown mistral model tokenizer")


_mapping: dict[str, str] = {
    "open-mistral-7b": "tokenizer.model.v1",
    "open-mixtral-8x7b": "tokenizer.model.v1",
    "mistral-embed": "tokenizer.model.v1",
    "mistral-small": "mistral_instruct_tokenizer_240216.model.v2",
    "mistral-large": "mistral_instruct_tokenizer_240216.model.v2",
    "open-mixtral-8x22b": "mistral_instruct_tokenizer_240323.model.v3",
    "codestral-22b": "mistral_instruct_tokenizer_240323.model.v3",
}
