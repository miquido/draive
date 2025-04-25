from collections.abc import Sequence

from haiway import MISSING, Missing, State

__all__ = (
    "BedrockChatConfig",
    "BedrockInputGuardraisConfig",
    "BedrockOutputGuardraisConfig",
)


class BedrockChatConfig(State):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = MISSING
    max_tokens: int | Missing = MISSING
    timeout: float | Missing = MISSING
    stop_sequences: Sequence[str] | Missing = MISSING


class BedrockInputGuardraisConfig(State):
    guardrail_identifier: str
    guardrail_version: str


class BedrockOutputGuardraisConfig(State):
    guardrail_identifier: str
    guardrail_version: str
