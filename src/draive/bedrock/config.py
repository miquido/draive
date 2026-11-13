from collections.abc import Sequence

from haiway import Configuration, Missing

__all__ = (
    "BedrockChatConfig",
    "BedrockInputGuardraisConfig",
    "BedrockOutputGuardraisConfig",
)


class BedrockChatConfig(Configuration):
    model: str
    temperature: float = 1.0
    top_p: float | Missing = Missing
    max_output_tokens: int | Missing = Missing
    stop_sequences: Sequence[str] | Missing = Missing
    # a guardrail enforced inline by the generation request itself, which avoids the
    # extra round trip of verifying the content through a separate call
    guardrail_identifier: str | Missing = Missing
    guardrail_version: str | Missing = Missing


class BedrockInputGuardraisConfig(Configuration):
    guardrail_identifier: str
    guardrail_version: str


class BedrockOutputGuardraisConfig(Configuration):
    guardrail_identifier: str
    guardrail_version: str
