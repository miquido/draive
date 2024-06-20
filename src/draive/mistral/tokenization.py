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
    model_file: str = _mapping.get(model_name, "mistral_v3_tokenizer.model")
    return SentencePieceProcessor(model_file=str(Path(__file__).parent / "tokens" / model_file))  # pyright: ignore[reportCallIssue]


_mapping: dict[str, str] = {
    "open-mistral-7b": "mistral_v1_tokenizer.model",
    "open-mixtral-8x7b": "mistral_v1_tokenizer.model",
    "mistral-embed": "mistral_v1_tokenizer.model",
    "mistral-small": "mistral_v2_tokenizer.model",
    "mistral-large": "mistral_v2_tokenizer.model",
    "open-mixtral-8x22b": "mistral_v3_tokenizer.model",
    "codestral-22b": "mistral_v3_tokenizer.model",
}
