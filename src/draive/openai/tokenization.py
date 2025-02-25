from typing import Any

from haiway import cache
from tiktoken import Encoding, encoding_for_model

from draive.openai.api import OpenAIAPI
from draive.tokenization import Tokenization

__all__ = [
    "OpenAITokenization",
]


class OpenAITokenization(OpenAIAPI):
    @cache(limit=2)
    async def tokenizer(
        self,
        model_name: str,
        /,
    ) -> Tokenization:
        """
        Prepare tokenizer for selected OpenAI model.
        """
        encoding: Encoding = _encoding(model_name=model_name)

        def openai_tokenize_text(
            text: str,
            **extra: Any,
        ) -> list[int]:
            return encoding.encode(text=text)

        return Tokenization(tokenize_text=openai_tokenize_text)


@cache(limit=4)
def _encoding(model_name: str) -> Encoding:
    return encoding_for_model(model_name=model_name)
