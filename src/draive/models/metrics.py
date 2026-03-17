from collections.abc import Sequence

from haiway import Missing, ObservabilityAttribute, ctx

from draive.models.types import ModelOutputSelection, ModelTools, ModelToolSpecification

__all__ = (
    "record_embedding_invocation",
    "record_embedding_metrics",
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
