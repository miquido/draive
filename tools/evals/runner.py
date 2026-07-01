from collections.abc import Callable, Sequence
from typing import cast

from haiway import Disposable, State, ctx, execute_concurrently

from draive.evaluation import EvaluatorResult
from draive.evaluators import cohen_kappa_evaluator

from .baseline import BaselineDocument, BaselineSample
from .registry import EvaluatorEntry, lookup_evaluator

__all__ = (
    "SampleFailure",
    "SampleOutcome",
    "SuiteResult",
    "run_suite",
)


class SampleOutcome(State):
    sample_id: str
    human_score: float
    evaluator_score: float


class SampleFailure(State):
    sample_id: str
    error: str


def _agreement_metric(
    result: EvaluatorResult,
    key: str,
    /,
) -> float:
    return float(cast(float, result.meta[key]))


class SuiteResult(State):
    evaluator: str
    outcomes: Sequence[SampleOutcome]
    agreement: EvaluatorResult
    failures: Sequence[SampleFailure] = ()

    def render(self) -> str:
        header: str = f"Human baseline verification: {self.evaluator}"
        lines: list[str] = [
            header,
            "=" * len(header),
            f"samples: {int(_agreement_metric(self.agreement, 'sample_count'))}",
            f"exact agreement: {_agreement_metric(self.agreement, 'exact_agreement') * 100:.1f}%",
            f"Cohen's kappa (nominal):            "
            f"{_agreement_metric(self.agreement, 'cohen_kappa'):.3f}",
            f"Cohen's kappa (quadratic weighted): "
            f"{_agreement_metric(self.agreement, 'quadratic_weighted_kappa'):.3f}",
            "",
            "interpretation (Landis & Koch, 1977):",
            "  < 0.00 poor | 0.00-0.20 slight | 0.21-0.40 fair",
            "  0.41-0.60 moderate | 0.61-0.80 substantial | 0.81-1.00 almost perfect",
            "",
            "per-sample comparison (human -> evaluator):",
        ]
        lines.extend(
            f"  {outcome.sample_id}: {outcome.human_score:.2f} -> {outcome.evaluator_score:.2f}"
            for outcome in self.outcomes
        )

        if self.failures:
            lines.append("")
            lines.append("failures:")
            lines.extend(f"  {failure.sample_id}: {failure.error}" for failure in self.failures)

        return "\n".join(lines)


async def _evaluate_sample(
    sample: BaselineSample,
    entry: EvaluatorEntry,
    /,
) -> SampleOutcome:
    if "evaluated" not in sample.inputs:
        raise ValueError(f"Sample '{sample.id}' missing required 'evaluated' input")

    result: EvaluatorResult = await entry.evaluator(
        sample.inputs["evaluated"],
        **entry.build_kwargs(sample.inputs),
    )

    if "error" in result.meta:
        # Evaluator swallowed an internal failure (e.g. a judge API error) and
        # returned a score of 0.0 instead of raising - surface it as a sample
        # failure rather than silently treating that placeholder as a real judgment.
        raise RuntimeError(f"Evaluator '{entry.name}' reported an error: {result.meta['error']}")

    return SampleOutcome(
        sample_id=sample.id,
        human_score=sample.human_score,
        evaluator_score=result.score,
    )


async def run_suite(
    document: BaselineDocument,
    /,
    *,
    config_state: Sequence[State],
    disposable_factories: Sequence[Callable[[], Disposable]] = (),
    concurrency: int = 4,
    continue_on_error: bool = False,
) -> SuiteResult:
    entry: EvaluatorEntry = lookup_evaluator(document.evaluator)

    async def evaluate(sample: BaselineSample) -> SampleOutcome:
        return await _evaluate_sample(sample, entry)

    async with ctx.scope(
        f"verify.{document.evaluator}",
        *config_state,
        disposables=tuple(factory() for factory in disposable_factories),
    ):
        raw: Sequence[SampleOutcome | BaseException]
        if continue_on_error:
            raw = await execute_concurrently(
                evaluate,
                document.samples,
                concurrent_tasks=max(1, concurrency),
                return_exceptions=True,
            )

        else:
            raw = await execute_concurrently(
                evaluate,
                document.samples,
                concurrent_tasks=max(1, concurrency),
            )

    outcomes: list[SampleOutcome] = []
    failures: list[SampleFailure] = []
    for sample, item in zip(document.samples, raw, strict=True):
        if isinstance(item, BaseException):
            failures.append(SampleFailure(sample_id=sample.id, error=str(item)))
            ctx.log_error(
                f"Sample '{sample.id}' failed during evaluation",
                exception=item,
            )

        else:
            outcomes.append(item)

    if not outcomes:
        raise RuntimeError(
            f"All {len(document.samples)} samples failed for evaluator '{document.evaluator}'"
        )

    outcomes.sort(key=lambda outcome: outcome.sample_id)

    # Measure evaluator <-> human agreement with the shipped framework evaluator.
    agreement: EvaluatorResult = await cohen_kappa_evaluator(
        tuple(outcome.evaluator_score for outcome in outcomes),
        reference=tuple(outcome.human_score for outcome in outcomes),
    )

    return SuiteResult(
        evaluator=document.evaluator,
        outcomes=tuple(outcomes),
        agreement=agreement,
        failures=tuple(failures),
    )
