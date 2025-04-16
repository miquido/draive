from pathlib import Path

from draive.gemini.api import GeminiAPI
from draive.sentencepiece import sentencepiece_processor, sentencepiece_tokenizer
from draive.tokenization import Tokenization

__all__ = ("GeminiTokenization",)


class GeminiTokenization(GeminiAPI):
    async def tokenizer(
        self,
        model_name: str,
        /,
    ) -> Tokenization:
        """
        Prepare tokenizer for selected Gemini model.
        """

        return Tokenization(
            text_tokenizing=sentencepiece_tokenizer(
                await sentencepiece_processor(model_path=_model_path(model_name))
            )
        )


def _model_path(model_name: str) -> str:
    model_file: str = _mapping.get(model_name, "gemini_tokenizer.model")
    return str(Path(__file__).parent / "tokens" / model_file)


_mapping: dict[str, str] = {}  # empty, using only default tokenizer
