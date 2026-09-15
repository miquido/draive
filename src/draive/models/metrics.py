import random

from haiway import Missing, ObservabilityAttribute, ctx

from draive.models.types import (
    ModelOutputSelection,
    ModelQuotaLimit,
    ModelRateLimit,
    ModelTools,
    ModelToolSpecification,
)

__all__ = (
    "model_rate_limit",
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
) -> None:
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


def record_usage_metrics(
    *,
    provider: str,
    model: str,
    input_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    output_tokens: int | None = None,
    reasoning_output_tokens: int | None = None,
) -> None:
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
