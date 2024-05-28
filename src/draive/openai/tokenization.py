from tiktoken import Encoding, encoding_for_model

from draive.openai.config import OpenAIChatConfig
from draive.scope import ctx
from draive.utils import Missing, cache, not_missing

__all__ = [
    "openai_tokenize_text",
]


def openai_tokenize_text(
    text: str,
) -> list[int]:
    model_name: str | Missing = ctx.state(OpenAIChatConfig).model
    if not_missing(model_name):
        return _encoding(model_name=model_name).encode(text=text)

    else:
        raise ValueError("Missing model name in OpenAIChatConfig")


@cache(limit=8)
def _encoding(model_name: str) -> Encoding:
    return encoding_for_model(model_name=model_name)
