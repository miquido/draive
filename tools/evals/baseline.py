from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml
from haiway import Meta, State

from draive.evaluation.value import evaluation_score_value

__all__ = (
    "BaselineDocument",
    "BaselineSample",
    "load_baseline",
)


class BaselineSample(State):
    id: str
    human_score: float
    inputs: Mapping[str, Any]
    meta: Meta = Meta.empty


class BaselineDocument(State):
    evaluator: str
    samples: Sequence[BaselineSample]
    description: str | None = None
    meta: Meta = Meta.empty


def _coerce_score(value: Any, /) -> float:
    match value:
        case bool() | int() | float():
            return evaluation_score_value(value)

        case str() as text:
            match text.lower().strip():
                case (
                    "none"
                    | "poor"
                    | "fair"
                    | "good"
                    | "excellent"
                    | "perfect"
                ) as normalized:
                    return evaluation_score_value(normalized)

                case _:
                    raise ValueError(f"Unsupported human_score value: {value!r}")

        case _:
            raise ValueError(f"Unsupported human_score value: {value!r}")


def _coerce_meta(value: Any, label: str, /) -> Meta:
    if value is None:
        return Meta.empty

    if not isinstance(value, Mapping):
        raise ValueError(f"'{label}' must be a mapping if provided")

    return Meta.of(cast(Mapping[str, Any], value))


def _coerce_sample(value: Any, index: int, /) -> BaselineSample:
    if not isinstance(value, Mapping):
        raise ValueError(f"Sample #{index}: must be a mapping")

    raw: Mapping[str, Any] = cast(Mapping[str, Any], value)

    if "human_score" not in raw:
        raise ValueError(f"Sample #{index}: missing required key 'human_score'")

    inputs_value: Any = raw.get("inputs")
    if not isinstance(inputs_value, Mapping):
        raise ValueError(f"Sample #{index}: missing or invalid 'inputs' mapping")

    inputs: Mapping[str, Any] = cast(Mapping[str, Any], inputs_value)

    return BaselineSample(
        id=str(raw.get("id") or f"sample-{index}"),
        human_score=_coerce_score(raw["human_score"]),
        inputs={str(key): inputs[key] for key in inputs},
        meta=_coerce_meta(raw.get("meta"), f"Sample #{index} 'meta'"),
    )


def load_baseline(path: Path | str, /) -> BaselineDocument:
    file_path: Path = Path(path)
    with file_path.open(encoding="utf-8") as handle:
        document_value: Any = yaml.safe_load(handle)

    if not isinstance(document_value, Mapping):
        raise ValueError(
            f"Baseline file '{file_path}' must contain a YAML mapping at the top level"
        )

    document: Mapping[str, Any] = cast(Mapping[str, Any], document_value)

    evaluator: Any = document.get("evaluator")
    if not isinstance(evaluator, str) or not evaluator:
        raise ValueError(f"Baseline file '{file_path}' missing required 'evaluator' key")

    raw_samples: Any = document.get("samples")
    if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, str) or not raw_samples:
        raise ValueError(f"Baseline file '{file_path}' must define a non-empty 'samples' list")

    description: Any = document.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError(f"'description' in '{file_path}' must be a string if provided")

    return BaselineDocument(
        evaluator=evaluator,
        samples=tuple(
            _coerce_sample(sample, index)
            for index, sample in enumerate(cast(Sequence[Any], raw_samples))
        ),
        description=description,
        meta=_coerce_meta(document.get("metadata"), f"'{file_path}' 'metadata'"),
    )
