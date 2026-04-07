from collections.abc import Callable
from typing import cast, overload

from google.genai.types import SpeechConfigDict
from haiway import MISSING, Missing

from draive.gemini.config import GeminiConfig

__all__ = ("speech_config", "unwrap_missing")


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
