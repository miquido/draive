from typing import Any

from tiktoken import Encoding, encoding_for_model

from draive.openai.config import OpenAIChatConfig
from draive.scope import ctx
from draive.utils import cache

__all__ = [
    "openai_tokenize_text",
]


def openai_tokenize_text(
    text: str,
    **extra: Any,
) -> list[int]:
    return _encoding(model_name=ctx.state(OpenAIChatConfig).model).encode(text=text)


@cache(limit=4)
def _encoding(model_name: str) -> Encoding:
    return encoding_for_model(model_name=model_name)
