from collections.abc import Callable
from typing import Any, cast, overload

from google.genai.errors import APIError
from google.genai.types import SpeechConfigDict, ThinkingConfigDict, ThinkingLevel
from haiway import MISSING, BasicValue, Missing

from draive.gemini.config import GeminiConfig

__all__ = (
    "RATE_LIMIT_STATUS_CODE",
    "combined_input_tokens",
    "quota_limit",
    "speech_config",
    "thinking_config",
    "unwrap_missing",
)

RATE_LIMIT_STATUS_CODE = 429


@overload
def unwrap_missing[Value](
    value: Value | Missing,
    /,
    default: Value,
) -> Value: ...


@overload
def unwrap_missing[Value](
    value: Value | Missing,
    /,
    default: Value | None = None,
) -> Value | None: ...


@overload
def unwrap_missing[Value, Result](
    value: Value | Missing,
    /,
    default: Value,
    *,
    transform: Callable[[Value], Result],
) -> Result: ...


@overload
def unwrap_missing[Value, Result](
    value: Value | Missing,
    /,
    default: Value | None = None,
    *,
    transform: Callable[[Value], Result],
) -> Result | None: ...


def unwrap_missing[Value, Result](
    value: Value | Missing,
    /,
    default: Result | Value | None = None,
    *,
    transform: Callable[[Value], Result] | None = None,
) -> Result | Value | None:
    if value is MISSING:
        return default

    elif transform is not None:
        return transform(cast(Value, value))

    else:
        return cast(Result, value)


def combined_input_tokens(
    prompt_tokens: int | None,
    tool_use_prompt_tokens: int | None,
    /,
) -> int | None:
    # tool use prompt tokens are reported separately from the prompt tokens
    if prompt_tokens is None:
        return tool_use_prompt_tokens

    if tool_use_prompt_tokens is None:
        return prompt_tokens

    return prompt_tokens + tool_use_prompt_tokens


def speech_config(
    config: GeminiConfig,
    /,
) -> SpeechConfigDict | None:
    if config.speech_voice_name is MISSING and config.speech_language_code is MISSING:
        return None

    speech_config: SpeechConfigDict = {}
    if config.speech_voice_name is not MISSING:
        speech_config["voice_config"] = {
            "prebuilt_voice_config": {
                "voice_name": cast(str, config.speech_voice_name),
            },
        }

    if config.speech_language_code is not MISSING:
        speech_config["language_code"] = cast(str, config.speech_language_code)

    return speech_config


def thinking_config(
    config: GeminiConfig,
    /,
) -> ThinkingConfigDict | None:
    # thinking level is the current API, thinking budget is kept for older models
    if config.thinking_level is not MISSING:
        return {
            "include_thoughts": True,
            "thinking_level": _resolve_thinking_level(cast(str, config.thinking_level)),
        }

    if config.thinking_budget is not MISSING:
        return {
            "include_thoughts": True,
            "thinking_budget": cast(int, config.thinking_budget),
        }

    return None


def _resolve_thinking_level(
    level: str,
    /,
) -> ThinkingLevel:
    match level:
        case "minimal":
            return ThinkingLevel.MINIMAL

        case "low":
            return ThinkingLevel.LOW

        case "medium":
            return ThinkingLevel.MEDIUM

        case "high":
            return ThinkingLevel.HIGH

        case _:
            raise ValueError(f"Unsupported thinking level: {level}")


def quota_limit(error: APIError) -> int | None:
    """Read the smallest enforced capacity from a Gemini quota failure.

    Parameters
    ----------
    error
        SDK error containing decoded JSON details.

    Returns
    -------
    int | None
        Smallest explicitly reported nonnegative quota, or None when unavailable.
        Missing values are not interpreted as zero.
    """
    # The SDK exposes decoded JSON as Any; only structured QuotaFailure entries
    # establish capacity. Missing quotaValue must never be interpreted as zero.
    details: BasicValue = cast(Any, error).details
    match details:
        case {"error": {"details": list() as entries}} | {"details": list() as entries}:
            pass

        case _:
            return None

    limit: int | None = None
    for entry in entries:
        match entry:
            case {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure" | "google.rpc.QuotaFailure",
                "violations": list() as violations,
            }:
                for violation in violations:
                    match violation:
                        case {"quotaValue": str() | int() as value} if not isinstance(value, bool):
                            try:
                                resolved = int(value)

                            except ValueError:
                                continue

                            if resolved >= 0:
                                limit = resolved if limit is None else min(limit, resolved)

                        case _:
                            continue

            case _:
                continue

    return limit
