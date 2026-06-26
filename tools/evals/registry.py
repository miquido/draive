from collections.abc import Mapping
from typing import Any, Final

from haiway import State

from draive.evaluation import Evaluator
from draive.evaluators import (
    coherence_evaluator,
    completeness_evaluator,
    conciseness_evaluator,
    consistency_evaluator,
    coverage_evaluator,
    creativity_evaluator,
    expectations_evaluator,
    factual_accuracy_evaluator,
    fluency_evaluator,
    groundedness_evaluator,
    helpfulness_evaluator,
    jailbreak_evaluator,
    readability_evaluator,
    relevance_evaluator,
    safety_evaluator,
    similarity_evaluator,
    tone_style_evaluator,
    truthfulness_evaluator,
)

__all__ = (
    "EvaluatorEntry",
    "available_evaluators",
    "lookup_evaluator",
)


class EvaluatorEntry(State):
    name: str
    evaluator: Evaluator[Any, ...]
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    description: str

    def build_kwargs(self, inputs: Mapping[str, Any], /) -> dict[str, Any]:
        missing: list[str] = [key for key in self.required_inputs if key not in inputs]
        if missing:
            raise ValueError(
                f"Evaluator '{self.name}' missing required input(s): {', '.join(missing)}"
            )

        # 'evaluated' is passed positionally by the runner.
        kwargs: dict[str, Any] = {
            key: inputs[key] for key in self.required_inputs if key != "evaluated"
        }
        for key in self.optional_inputs:
            value: Any = inputs.get(key)
            if value is not None:
                kwargs[key] = value

        return kwargs


def _entry(
    evaluator: Evaluator[Any, ...],
    /,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = ("guidelines",),
    description: str,
) -> EvaluatorEntry:
    return EvaluatorEntry(
        name=evaluator.name,
        evaluator=evaluator,
        required_inputs=("evaluated", *required),
        optional_inputs=optional,
        description=description,
    )


_ENTRIES: Final[tuple[EvaluatorEntry, ...]] = (
    _entry(
        coherence_evaluator,
        required=("reference",),
        description="Structural coherence of content given a reference.",
    ),
    _entry(
        completeness_evaluator,
        required=("user_query",),
        description="Whether content fully addresses every part of a user query.",
    ),
    _entry(
        conciseness_evaluator,
        required=("reference",),
        description="Compactness of content relative to a reference.",
    ),
    _entry(
        consistency_evaluator,
        required=("reference",),
        description="Factual/logical consistency between content and reference.",
    ),
    _entry(
        coverage_evaluator,
        required=("reference",),
        description="Extent to which content covers reference material.",
    ),
    _entry(
        creativity_evaluator,
        required=(),
        description="Originality and creative thinking in content (no reference).",
    ),
    _entry(
        expectations_evaluator,
        required=("expectations",),
        description="Fulfilment of explicit expectations stated alongside the content.",
    ),
    _entry(
        factual_accuracy_evaluator,
        required=(),
        optional=("guidelines", "reference"),
        description="Factual correctness judged against a reference when provided, "
        "else world knowledge.",
    ),
    _entry(
        fluency_evaluator,
        required=(),
        description="Linguistic fluency of content.",
    ),
    _entry(
        groundedness_evaluator,
        required=("reference",),
        description="Whether claims in content are grounded in the reference.",
    ),
    _entry(
        helpfulness_evaluator,
        required=("user_query",),
        description="How well content addresses user needs for a given query.",
    ),
    _entry(
        jailbreak_evaluator,
        required=(),
        description="Detection of attempts to bypass model safeguards.",
    ),
    _entry(
        readability_evaluator,
        required=(),
        description="Readability of content.",
    ),
    _entry(
        relevance_evaluator,
        required=("reference",),
        description="Topical relevance of content given a reference.",
    ),
    _entry(
        safety_evaluator,
        required=(),
        description="Presence of harmful or dangerous material.",
    ),
    _entry(
        similarity_evaluator,
        required=("reference",),
        description="Semantic similarity between content and reference (LLM judge).",
    ),
    _entry(
        tone_style_evaluator,
        required=("expected_tone_style",),
        description="Match between content tone/style and explicit expectations.",
    ),
    _entry(
        truthfulness_evaluator,
        required=(),
        optional=("guidelines", "reference"),
        description="Truthfulness of content; reference is supplemental, not required.",
    ),
)

_REGISTRY: Final[Mapping[str, EvaluatorEntry]] = {entry.name: entry for entry in _ENTRIES}


def available_evaluators() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def lookup_evaluator(name: str, /) -> EvaluatorEntry:
    try:
        return _REGISTRY[name]

    except KeyError as exc:
        raise KeyError(
            f"Unknown evaluator '{name}'. Available: {', '.join(available_evaluators())}"
        ) from exc
