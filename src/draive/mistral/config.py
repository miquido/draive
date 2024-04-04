from typing import Literal, TypedDict

from draive.helpers import getenv_float, getenv_int, getenv_str
from draive.types import State

__all__ = [
    "MistralChatConfig",
]


class ResponseFormat(TypedDict):
    type: Literal["text", "json_object"]


class MistralChatConfig(State):
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
    top_p: float | None = None
    seed: int | None = getenv_int("MISTRAL_SEED")
    max_tokens: int | None = None
    response_format: ResponseFormat | None = None
    timeout: float | None = None
    recursion_limit: int = 4

    def metric_summary(
        self,
        trimmed: bool,
    ) -> str:
        result: str = f"mistral config:\n+ model: {self.model}\n+ temperature: {self.temperature}"
        if self.top_p:
            result += f"\n+ top_p: {self.top_p}"
        if self.max_tokens:
            result += f"\n+ max_tokens: {self.max_tokens}"
        if self.seed:
            result += f"\n+ seed: {self.seed}"
        if self.response_format:
            result += f"\n+ response_format: {self.response_format.get('type', 'text')}"
        if self.timeout:
            result += f"\n+ timeout: {self.timeout}"
        result += f"\n+ recursion_limit: {self.recursion_limit}"

        return result.replace("\n", "\n|   ")
