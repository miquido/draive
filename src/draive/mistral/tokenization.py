from pathlib import Path
from typing import Any, cast

from sentencepiece import SentencePieceProcessor  # pyright: ignore[reportMissingTypeStubs]

from draive.mistral.config import MistralChatConfig
from draive.scope import ctx
from draive.utils import cache

__all__ = [
    "mistral_tokenize_text",
]


def mistral_tokenize_text(
    text: str,
    **extra: Any,
) -> list[int]:
    return cast(
        list[int],
        _encoding(model_name=ctx.state(MistralChatConfig).model).encode(text),  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    )


@cache(limit=4)
def _encoding(model_name: str) -> SentencePieceProcessor:
    model_file: str = _mapping.get(model_name, "mistral_instruct_tokenizer_240323.model.v3")
    return SentencePieceProcessor(model_file=str(Path(__file__).parent / "tokens" / model_file))  # pyright: ignore[reportCallIssue]


_mapping: dict[str, str] = {
    "open-mistral-7b": "tokenizer.model.v1",
    "open-mixtral-8x7b": "tokenizer.model.v1",
    "mistral-embed": "tokenizer.model.v1",
    "mistral-small": "mistral_instruct_tokenizer_240216.model.v2",
    "mistral-large": "mistral_instruct_tokenizer_240216.model.v2",
    "open-mixtral-8x22b": "mistral_instruct_tokenizer_240323.model.v3",
    "codestral-22b": "mistral_instruct_tokenizer_240323.model.v3",
}
