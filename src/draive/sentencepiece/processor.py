from typing import Any, cast

from haiway import asynchronous, cache
from sentencepiece import SentencePieceProcessor  # pyright: ignore[reportMissingTypeStubs]

from draive.tokenization import TextTokenizing

__all__ = (
    "sentencepiece_processor",
    "sentencepiece_tokenizer",
)


def sentencepiece_tokenizer(
    processor: SentencePieceProcessor,
    /,
) -> TextTokenizing:
    def sentencepiece_tokenize_text(
        text: str,
        **extra: Any,
    ) -> list[int]:
        return cast(
            list[int],
            processor.encode(text),  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        )

    return sentencepiece_tokenize_text


@asynchronous
@cache(limit=1)
def sentencepiece_processor(
    *,
    model_path: str,
) -> SentencePieceProcessor:
    return SentencePieceProcessor(model_file=model_path)  # pyright: ignore[reportCallIssue]
