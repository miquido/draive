from os import getenv
from typing import Literal

from openai.types.chat.completion_create_params import ResponseFormat

from draive.helpers import getenv_float, getenv_int
from draive.types import State

__all__ = [
    "OpenAIChatConfig",
    "OpenAIEmbeddingConfig",
    "OpenAIImageGenerationConfig",
]


class OpenAIChatConfig(State):
    model: (
        Literal[
            "gpt-4-0125-preview",
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-4-vision-preview",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613",
        ]
        | str
    ) = getenv("OPENAI_MODEL", default="gpt-3.5-turbo-0125")
    temperature: float = getenv_float("OPENAI_TEMPERATURE", 0.0)
    top_p: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None
    seed: int | None = getenv_int("OPENAI_SEED")
    response_format: ResponseFormat | None = None
    vision_details: Literal["auto", "low", "high"] = "auto"
    timeout: float | None = None
    recursion_limit: int = 4

    def metric_summary(
        self,
        trimmed: bool,
    ) -> str:
        result: str = f"openai config:\n+ model: {self.model}\n+ temperature: {self.temperature}"
        if self.top_p:
            result += f"\n+ top_p: {self.top_p}"
        if self.frequency_penalty:
            result += f"\n+ frequency_penalty: {self.frequency_penalty}"
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


class OpenAIEmbeddingConfig(State):
    model: (
        Literal[
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
        | str
    ) = "text-embedding-3-small"
    dimensions: int | None = None
    batch_size: int = 32
    encoding_format: Literal["float", "base64"] | None = None
    timeout: float | None = None

    def metric_summary(
        self,
        trimmed: bool,
    ) -> str:
        result: str = f"openai config\n+ model: {self.model}"
        if self.dimensions:
            result += f"\n+ dimensions: {self.dimensions}"
        result += f"\n+ batch_size: {self.batch_size}"
        if self.encoding_format:
            result += f"\n+ encoding_format: {self.encoding_format}"
        if self.timeout:
            result += f"\n+ timeout: {self.timeout}"

        return result.replace("\n", "\n|   ")


class OpenAIImageGenerationConfig(State):
    model: Literal["dall-e-2", "dall-e-3"] | str = "dall-e-2"
    quality: Literal["standard", "hd"] = "standard"
    size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"] = "1024x1024"
    style: Literal["vivid", "natural"] = "vivid"
    timeout: float | None = None
    response_format: Literal["url", "b64_json"] = "b64_json"

    def metric_summary(
        self,
        trimmed: bool,
    ) -> str:
        result: str = (
            f"openai config\n+ model: {self.model}"
            f"\n+ quality: {self.quality}"
            f"\n+ style: {self.style}"
            f"\n+ size: {self.size}"
            f"\n+ response_format: {self.response_format}"
        )
        if self.timeout:
            result += f"\n+ timeout: {self.timeout}"

        return result.replace("\n", "\n|   ")
