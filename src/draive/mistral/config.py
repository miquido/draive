from typing import Literal, TypedDict

from draive.helpers import MISSING, Missing, getenv_float, getenv_int, getenv_str
from draive.types import Model

__all__ = [
    "MistralChatConfig",
    "MistralEmbeddingConfig",
]


class ResponseFormat(TypedDict):
    type: Literal["text", "json_object"]


class MistralChatConfig(Model):
    model: (
        Literal[
            "mistral-large-latest",
            "mistral-large-2402",
            "mistral-medium-latest",
            "mistral-medium-2312",
            "mistral-small-latest",
            "mistral-small-2402",
            "mistral-small-2312",
            "open-mixtral-8x7b",
            "open-mistral-7b",
            "mistral-tiny-2312",
        ]
        | str
    ) = getenv_str("MISTRAL_MODEL", default="open-mistral-7b")
    temperature: float = getenv_float("MISTRAL_TEMPERATURE", 0.0)
    top_p: float | Missing = MISSING
    seed: int | Missing = getenv_int("MISTRAL_SEED") or MISSING
    max_tokens: int | Missing = MISSING
    response_format: ResponseFormat | Missing = MISSING
    timeout: float | Missing = MISSING
    recursion_limit: int = 4


class MistralEmbeddingConfig(Model):
    model: Literal["mistral-embed"] | str = "mistral-embed"
    batch_size: int = 32
