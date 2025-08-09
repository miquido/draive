from collections.abc import Sequence

from haiway import MISSING, Configuration, Missing

__all__ = (
    "BedrockChatConfig",
    "BedrockInputGuardraisConfig",
    "BedrockOutputGuardraisConfig",
)


class BedrockChatConfig(Configuration):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    max_output_tokens: int | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class BedrockInputGuardraisConfig(Configuration):
    guardrail_identifier: str
    guardrail_version: str


class BedrockOutputGuardraisConfig(Configuration):
    guardrail_identifier: str
    guardrail_version: str
