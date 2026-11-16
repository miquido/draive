import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import cast

from haiway import Missing
from mistralai.client import UNSET
from mistralai.client.errors import MistralError
from mistralai.client.types.basemodel import Unset

from draive.models import ModelOutputFailed

__all__ = (
    "mistral_error_reason",
    "mistral_failures",
    "unwrap_missing_list_to_unset",
    "unwrap_missing_to_unset",
)


def unwrap_missing_to_unset[Value](
    value: Value | Missing,
    /,
) -> Value | Unset:
    if value is Missing:
        return UNSET
    else:
        return cast(Value, value)


def unwrap_missing_list_to_unset[Value](
    value: Sequence[Value] | Missing,
    /,
) -> list[Value] | Unset:
    if value is Missing:
        return UNSET
    else:
        return list(value)


RATE_LIMIT_STATUS_CODE: int = 429
_MAX_DETAIL_LENGTH: int = 300


def mistral_error_reason(
    error: MistralError,
    /,
) -> str:
    """Describe a Mistral HTTP error without exposing the response body.

    Parameters
    ----------
    error : MistralError
        Error raised by the Mistral SDK.

    Returns
    -------
    str
        HTTP status followed by the short top-level message reported by the provider,
        when there is one. The raw response body is never included.
    """
    reason: str = f"HTTP {error.status_code}"
    detail: str | None = _error_detail(error.body)
    if detail:
        return f"{reason} - {detail}"

    return reason


@contextmanager
def mistral_failures(
    *,
    model: str,
) -> Generator[None]:
    """Translate Mistral HTTP errors into a typed exception free of the response body.

    Rate limiting errors are passed through untouched, so they can be mapped by the caller.

    Parameters
    ----------
    model : str
        Model identifier used for the request.

    Raises
    ------
    ModelOutputFailed
        When the wrapped call fails with a Mistral error other than rate limiting.
    """
    try:
        yield

    except MistralError as exc:
        if exc.status_code == RATE_LIMIT_STATUS_CODE:
            raise

        raise ModelOutputFailed(
            provider="mistral",
            model=model,
            reason=mistral_error_reason(exc),
        ) from exc


def _error_detail(
    body: str | None,
    /,
) -> str | None:
    if not body:
        return None

    try:
        parsed: object = json.loads(body)

    except ValueError:
        return None

    if not isinstance(parsed, dict):
        return None

    fields: dict[str, object] = cast(dict[str, object], parsed)
    for key in ("message", "error", "detail"):
        value: object = fields.get(key)
        if isinstance(value, str):
            collapsed: str = " ".join(value.split())
            if not collapsed:
                continue

            if len(collapsed) > _MAX_DETAIL_LENGTH:
                return f"{collapsed[: _MAX_DETAIL_LENGTH - 1]}…"

            return collapsed

    return None
