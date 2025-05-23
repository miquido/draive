from collections.abc import Sequence

from haiway import MISSING, Missing

from draive.configuration import Config

__all__ = (
    "BedrockChatConfig",
    "BedrockInputGuardraisConfig",
    "BedrockOutputGuardraisConfig",
)


class BedrockChatConfig(Config):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    max_tokens: int | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class BedrockInputGuardraisConfig(Config):
    guardrail_identifier: str
    guardrail_version: str


class BedrockOutputGuardraisConfig(Config):
    guardrail_identifier: str
    guardrail_version: str
