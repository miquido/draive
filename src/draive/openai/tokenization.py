from typing import Any

from haiway import cache
from tiktoken import Encoding, encoding_for_model

from draive.tokenization import TextTokenizing

__all__ = [
    "openai_text_tokenizer",
]


def openai_text_tokenizer(
    model_name: str,
    /,
) -> TextTokenizing:
    encoding: Encoding = _encoding(model_name=model_name)

    def openai_tokenize_text(
        text: str,
        **extra: Any,
    ) -> list[int]:
        return encoding.encode(text=text)

    return openai_tokenize_text


@cache(limit=4)
def _encoding(model_name: str) -> Encoding:
    return encoding_for_model(model_name=model_name)
