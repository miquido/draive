import random
from collections.abc import Coroutine, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import cast

from haiway import Missing, ObservabilityAttribute, ctx

from draive.models.types import (
    ModelException,
    ModelOutputFailed,
    ModelOutputSelection,
    ModelQuotaLimit,
    ModelRateLimit,
    ModelTools,
    ModelToolSpecification,
)
from draive.utils.concurrent import gather_all

__all__ = (
    "model_rate_limit",
    "provider_failure",
    "provider_gather",
    "provider_rate_limit",
    "record_embedding_invocation",
    "record_embedding_metrics",
    "record_guardrails_invocation",
    "record_model_invocation",
    "record_usage_metrics",
)


def record_model_invocation(
    *,
    provider: str,
    model: str,
    tools: ModelTools,
    output: ModelOutputSelection,
    **other: ObservabilityAttribute | Missing | None,
) -> None:
    """Record attributes describing a model invocation within the current scope.

    Parameters
    ----------
    provider : str
        Provider identifier used for the invocation.
    model : str
        Provider model identifier used for the invocation.
    tools : ModelTools
        Tools offered to the model, recorded by name together with the selection mode.
    output : ModelOutputSelection
        Requested output selection.
    **other : ObservabilityAttribute | Missing | None
        Additional provider specific attributes recorded with the ``model.`` prefix.
    """
    model_output: str
    if isinstance(output, type):
        model_output = f"state:{output.__name__}"

    else:
        model_output = str(output)

    model_tools_selection: str
    if isinstance(tools.selection, ModelToolSpecification):
        model_tools_selection = f"tool:{tools.selection.name}"

    else:
        model_tools_selection = tools.selection

    ctx.record_info(
        attributes={
            "model.provider": provider,
            "model.name": model,
            "model.tools": [tool.name for tool in tools.specification],
            "model.tools.selection": model_tools_selection,
            "model.output": model_output,
            **{f"model.{key}": value for key, value in other.items()},
        }
    )


def model_rate_limit(
    *,
    provider: str,
    model: str,
    retry_after: str | float | None,
    error_code: str | None = None,
    quota_limit: int | None = None,
) -> ModelRateLimit | ModelQuotaLimit:
    """Record a rate limit event and build the matching exception.

    Every provider reports rate limiting differently, recording it through a single
    helper keeps the observed attributes identical regardless of which one applied it.

    Parameters
    ----------
    provider : str
        Provider identifier that applied the limit.
    model : str
        Provider model identifier affected by the limit.
    retry_after : str | float | None
        Delay before a retry, either already resolved or as reported by the provider.
        A missing or unparseable value falls back to a randomized short delay, which
        spreads retries of concurrent requests instead of aligning them.
    error_code : str | None, default=None
        Structured provider error code, used to recognize exhausted quota.
    quota_limit : int | None, default=None
        Enforced capacity from structured quota failure details, when available.
        An explicit zero is a quota failure regardless of a reported retry delay.

    Returns
    -------
    ModelRateLimit | ModelQuotaLimit
        Exception to raise for the recorded limit. Explicit quota failures have no
        retry delay and are not subclasses of ``ModelRateLimit``.
    """
    reason: str | None = None
    if quota_limit == 0:
        reason = "Provider reported zero quota capacity"

    elif error_code in ("insufficient_quota", "billing_hard_limit_reached"):
        reason = f"Provider reported {error_code}"

    if reason is not None:
        ctx.record_warning(
            event="model.quota_limit",
            attributes={
                "model.provider": provider,
                "model.name": model,
                "model.quota_limit.reason": reason,
            },
        )
        return ModelQuotaLimit(
            provider=provider,
            model=model,
            reason=reason,
        )

    delay: float | None
    match retry_after:
        case float() as retry_delay:
            delay = retry_delay

        case int() as convertable:
            delay = float(convertable)

        case str() as described:
            try:
                delay = float(described)

            except ValueError:
                delay = None

        case _:
            delay = None

    ctx.record_warning(
        event="model.rate_limit",
        attributes={
            "model.provider": provider,
            "model.name": model,
            "model.retry_after": delay,
        },
    )

    return ModelRateLimit(
        provider=provider,
        model=model,
        retry_after=delay if delay is not None else random.uniform(0.3, 3.0),  # nosec: B311
    )


def record_guardrails_invocation(
    *,
    provider: str,
    model: str | None = None,
    **other: ObservabilityAttribute,
) -> None:
    """Record attributes describing a guardrails provider invocation.

    Parameters
    ----------
    provider : str
        Provider identifier used for the invocation.
    model : str | None, default=None
        Provider model identifier, when the provider uses a model.
    **other : ObservabilityAttribute
        Additional attributes recorded with the ``guardrails.`` prefix.
    """
    ctx.record_info(
        attributes={
            "guardrails.provider": provider,
            "guardrails.model": model,
            **{f"guardrails.{key}": value for key, value in other.items()},
        }
    )


def record_embedding_invocation(
    *,
    provider: str,
    model: str,
    embedding_type: str,
    batch_size: int,
    **other: ObservabilityAttribute,
) -> None:
    """Record attributes describing an embedding invocation within the current scope.

    Parameters
    ----------
    provider : str
        Provider identifier used for the invocation.
    model : str
        Provider model identifier used for the invocation.
    embedding_type : str
        Kind of embedded values, i.e. ``"text"`` or ``"image"``.
    batch_size : int
        Maximum number of values sent within a single request.
    **other : ObservabilityAttribute
        Additional attributes recorded with the ``embedding.`` prefix.
    """
    ctx.record_info(
        attributes={
            "embedding.provider": provider,
            "embedding.model": model,
            "embedding.type": embedding_type,
            "embedding.batch_size": batch_size,
            **{f"embedding.{key}": value for key, value in other.items()},
        }
    )


def record_embedding_metrics(
    *,
    provider: str,
    model: str,
    embedding_type: str,
    items: int | None = None,
    batches: int | None = None,
    input_tokens: int | None = None,
) -> None:
    """Record counters describing the work done by an embedding request.

    Parameters
    ----------
    provider : str
        Provider identifier used for the request.
    model : str
        Provider model identifier used for the request.
    embedding_type : str
        Kind of embedded values, i.e. ``"text"`` or ``"image"``.
    items : int | None, default=None
        Number of embedded values.
    batches : int | None, default=None
        Number of requests the values were split into.
    input_tokens : int | None, default=None
        Number of input tokens reported by the provider.
    """
    attributes: dict[str, str] = {
        "embedding.provider": provider,
        "embedding.model": model,
        "embedding.type": embedding_type,
    }

    if items is not None:
        ctx.record_info(
            metric="embedding.items",
            value=items,
            unit="count",
            kind="counter",
            attributes=attributes,
        )

    if batches is not None:
        ctx.record_info(
            metric="embedding.batches",
            value=batches,
            unit="count",
            kind="counter",
            attributes=attributes,
        )

    if input_tokens is not None:
        ctx.record_info(
            metric="embedding.input_tokens",
            value=input_tokens,
            unit="tokens",
            kind="counter",
            attributes=attributes,
        )


def record_usage_metrics(
    *,
    provider: str,
    model: str,
    input_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    output_tokens: int | None = None,
    reasoning_output_tokens: int | None = None,
) -> None:
    """Record token usage counters of a model invocation.

    Parameters
    ----------
    provider : str
        Provider identifier used for the invocation.
    model : str
        Provider model identifier used for the invocation.
    input_tokens : int | None, default=None
        Number of input tokens.
    cached_input_tokens : int | None, default=None
        Number of input tokens served from the provider cache.
    output_tokens : int | None, default=None
        Number of output tokens.
    reasoning_output_tokens : int | None, default=None
        Number of output tokens spent on reasoning.
    """
    attributes: dict[str, str] = {
        "model.provider": provider,
        "model.name": model,
    }

    if input_tokens is not None:
        ctx.record_info(
            metric="model.input_tokens",
            value=input_tokens,
            unit="tokens",
            kind="counter",
            attributes=attributes,
        )

    if cached_input_tokens is not None:
        ctx.record_info(
            metric="model.input_tokens.cached",
            value=cached_input_tokens,
            unit="tokens",
            kind="counter",
            attributes=attributes,
        )

    if output_tokens is not None:
        ctx.record_info(
            metric="model.output_tokens",
            value=output_tokens,
            unit="tokens",
            kind="counter",
            attributes=attributes,
        )

    if reasoning_output_tokens is not None:
        ctx.record_info(
            metric="model.output_tokens.reasoning",
            value=reasoning_output_tokens,
            unit="tokens",
            kind="counter",
            attributes=attributes,
        )


@contextmanager
def provider_failure(
    *,
    provider: str,
    model: str,
    operation: str,
) -> Generator[None]:
    """Translate unexpected provider SDK errors into a typed model exception.

    Wraps a single provider call outside of the chat completion flow, i.e. embedding,
    moderation or image generation. Model exceptions already raised within, rate limits
    among them, are passed through untouched and cancellation is never intercepted.

    Parameters
    ----------
    provider : str
        Provider identifier used for the request.
    model : str
        Provider model identifier used for the request.
    operation : str
        Short operation name included in the error, i.e. ``"embedding"``.

    Raises
    ------
    ModelRateLimit
        When the provider answers with a rate limiting status.
    ModelOutputFailed
        When the wrapped call fails with any other exception.
    """
    try:
        yield

    except ModelException:
        raise

    except Exception as exc:
        rate_limit = provider_rate_limit(provider=provider, model=model, error=exc)
        if rate_limit is not None:
            raise rate_limit from exc

        ctx.log_error(
            f"{provider} {operation} request failed",
            exception=exc,
        )
        raise ModelOutputFailed(
            provider=provider,
            model=model,
            reason=f"{operation} request failed - {type(exc).__name__}: {exc}",
        ) from exc


async def provider_gather[Result](
    *awaitables: Coroutine[None, None, Result],
    provider: str,
    model: str,
    operation: str,
) -> Sequence[Result]:
    """Run provider calls concurrently translating failures into typed exceptions.

    The remaining calls are cancelled when one of them fails.

    Parameters
    ----------
    *awaitables : Coroutine[None, None, Result]
        Provider calls to run concurrently.
    provider : str
        Provider identifier used for the requests.
    model : str
        Provider model identifier used for the requests.
    operation : str
        Short operation name included in the error, i.e. ``"embedding"``.

    Returns
    -------
    Sequence[Result]
        Results in the order of the given calls.

    Raises
    ------
    ModelOutputFailed
        When any of the calls fails with an unexpected exception.
    """
    with provider_failure(provider=provider, model=model, operation=operation):
        return await gather_all(*awaitables)


_RATE_LIMIT_STATUS: int = 429


def provider_rate_limit(
    *,
    provider: str,
    model: str,
    error: Exception,
) -> ModelRateLimit | ModelQuotaLimit | None:
    """Recognize a rate limiting answer of a provider SDK.

    Provider SDKs share no common error type, they all expose the status of a failed
    request though, which is what is used to recognize the limit.

    Parameters
    ----------
    provider : str
        Provider identifier used for the request.
    model : str
        Provider model identifier used for the request.
    error : Exception
        Error raised by the provider SDK.

    Returns
    -------
    ModelRateLimit | ModelQuotaLimit | None
        Exception to raise when the error is a rate limit, otherwise ``None``.
    """
    rate_limited, retry_after = _rate_limiting(error)
    if not rate_limited:
        return None

    return model_rate_limit(
        provider=provider,
        model=model,
        retry_after=retry_after,
    )


def _rate_limiting(
    error: Exception,
    /,
) -> tuple[bool, str | None]:
    # provider SDKs share no common error type, they all expose the status of a failed
    # request, as `status_code` or `code`, and the response headers directly or by response
    status: object = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)

    if status != _RATE_LIMIT_STATUS:
        return False, None

    headers: object = getattr(error, "headers", None)
    if headers is None:
        headers = getattr(getattr(error, "response", None), "headers", None)

    if isinstance(headers, Mapping):
        for key, value in cast(Mapping[object, object], headers).items():
            if str(key).lower() == "retry-after" and isinstance(value, str):
                return True, value

    return True, None
