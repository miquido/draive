from typing import Any, cast

from sentencepiece import SentencePieceProcessor  # pyright: ignore[reportMissingTypeStubs]

from draive.scope import ctx
from draive.sentencepiece.config import SentencePieceConfig
from draive.utils import cache, not_missing

__all__ = [
    "sentencepiece_tokenize_text",
]


def sentencepiece_tokenize_text(
    text: str,
    **extra: Any,
) -> list[int]:
    config: SentencePieceConfig = ctx.state(SentencePieceConfig).updated(**extra)
    if not_missing(config.model_path):
        return cast(
            list[int],
            _encoding(model_path=config.model_path).encode(text),  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        )
    else:
        raise ValueError("Missing sentencepiece tokenizer model path")


@cache(limit=4)
def _encoding(model_path: str) -> SentencePieceProcessor:
    return SentencePieceProcessor(model_file=model_path)  # pyright: ignore[reportCallIssue]
