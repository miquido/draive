from pathlib import Path

from draive.mistral.api import MistralAPI
from draive.sentencepiece import sentencepiece_processor, sentencepiece_tokenizer
from draive.tokenization import Tokenization

__all__ = ("MistralTokenization",)


class MistralTokenization(MistralAPI):
    async def tokenizer(
        self,
        model_name: str,
        /,
    ) -> Tokenization:
        """
        Prepare tokenizer for selected Mistral model.
        """

        return Tokenization(
            text_tokenizing=sentencepiece_tokenizer(
                await sentencepiece_processor(model_path=_model_path(model_name))
            )
        )


def _model_path(
    model_name: str,
    /,
) -> str:
    model_file: str = _mapping.get(model_name, "mistral_v3_tokenizer.model")
    return str(Path(__file__).parent / "tokens" / model_file)


_mapping: dict[str, str] = {
    "open-mistral-7b": "mistral_v1_tokenizer.model",
    "open-mixtral-8x7b": "mistral_v1_tokenizer.model",
    "mistral-embed": "mistral_v1_tokenizer.model",
    "mistral-small": "mistral_v2_tokenizer.model",
    "mistral-large": "mistral_v2_tokenizer.model",
    "open-mixtral-8x22b": "mistral_v3_tokenizer.model",
    "codestral-22b": "mistral_v3_tokenizer.model",
}
