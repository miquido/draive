import random
from collections.abc import Sequence

from haiway import Missing, ObservabilityAttribute, ctx

from draive.models.types import (
    ModelOutputSelection,
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
    temperature: float | Missing | None = None,
    max_output_tokens: int | Missing | None = None,
    stop_sequences: Sequence[str] | Missing | None = None,
    **other: ObservabilityAttribute,
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
            "model.temperature": temperature,
            "model.max_output_tokens": max_output_tokens,
            "model.tools": [tool.name for tool in tools.specification],
            "model.tools.selection": model_tools_selection,
            "model.output": model_output,
            "model.stop_sequences": stop_sequences,
            **{f"model.{key}": value for key, value in other.items()},
        }
    )


def model_rate_limit(
    *,
    provider: str,
    model: str,
    retry_after: str | float | None,
) -> ModelRateLimit:
    """Record a rate limit event and build the matching exception.

    Every provider reports rate limiting differently, recording it through a single
    helper keeps the observed attributes identical regardless of which one applied it.

    Parameters
    ----------
    provider
        Provider identifier that applied the limit.
    model
        Provider model identifier affected by the limit.
    retry_after
        Delay before a retry, either already resolved or as reported by the provider.
        A missing or unparseable value falls back to a randomized short delay, which
        spreads retries of concurrent requests instead of aligning them.

    Returns
    -------
    ModelRateLimit
        Exception to raise for the recorded limit.
    """
    delay: float
    match retry_after:
        case None:
            delay = random.uniform(0.3, 3.0)  # nosec: B311

        case float() | int() as resolved:
            delay = float(resolved)

        case reported:
            try:
                delay = float(reported)

            except ValueError:
                delay = random.uniform(0.3, 3.0)  # nosec: B311

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
        retry_after=delay,
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
