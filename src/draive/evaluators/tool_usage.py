from collections.abc import Mapping, Sequence
from typing import Self, final

from haiway import BasicValue, State

from draive.evaluation import EvaluationScore, evaluator
from draive.models import ModelContext, ModelOutput, ModelToolRequest

__all__ = (
    "ToolUsageRequirement",
    "tool_usage_context_evaluator",
)


@final
class ToolUsageRequirement(State):
    """
    Specification of a required tool invocation in a model context.

    Attributes
    ----------
    tool : str
        Name of the tool that must be invoked.
    arguments : Mapping[str, BasicValue] | None
        Optional subset of arguments that the matching tool call must contain.
        When provided, every listed key/value must be present in the call's
        arguments (extra arguments are allowed). When ``None``, any invocation
        of the tool satisfies the requirement.
    """

    @classmethod
    def of(
        cls,
        tool: str,
        /,
        *,
        arguments: Mapping[str, BasicValue] | None = None,
    ) -> Self:
        return cls(
            tool=tool,
            arguments=arguments,
        )

    tool: str
    arguments: Mapping[str, BasicValue] | None = None


@evaluator(name="tool_usage_context")
async def tool_usage_context_evaluator(
    evaluated: ModelContext,
    /,
    *,
    required: Sequence[ToolUsageRequirement | str] = (),
    expected: Sequence[str] = (),
    forbidden: Sequence[str] = (),
    strict: bool = True,
) -> EvaluationScore:
    """
    Evaluate tool usage within a model conversation context.

    Inspects all ``ModelToolRequest`` blocks present in the context and verifies
    that the conversation:

    * invokes every tool listed in ``required`` (optionally matching the
      specified argument subset);
    * invokes at least one tool from ``expected`` (when ``expected`` is non-empty);
    * does not invoke any tool listed in ``forbidden``.

    Parameters
    ----------
    evaluated : ModelContext
        Conversation context to inspect.
    required : Sequence[ToolUsageRequirement | str]
        Tools that must be invoked. Plain strings are normalized to
        ``ToolUsageRequirement`` without argument constraints.
    expected : Sequence[str]
        Set of tools where at least one must be invoked.
    forbidden : Sequence[str]
        Tools that must not be invoked.
    strict : bool
        When True, returns 1.0 only if every requirement is satisfied and 0.0
        otherwise. When False, returns the proportion of satisfied checks,
        where each entry in ``required`` counts as an individual check, while
        ``expected`` and ``forbidden`` are each evaluated as a single
        group-level check (regardless of how many entries they contain). The
        score is the number of satisfied checks divided by the total number
        of checks (``len(required) + 1`` if ``expected`` is non-empty
        ``+ 1`` if ``forbidden`` is non-empty).

    Returns
    -------
    EvaluationScore
        Tool usage evaluation result.
    """
    if not required and not expected and not forbidden:
        return EvaluationScore.of(
            0.0,
            meta={"comment": "No tool usage requirements were provided!"},
        )

    tool_requests: list[ModelToolRequest] = [
        block
        for element in evaluated
        if isinstance(element, ModelOutput)
        for block in element.output
        if isinstance(block, ModelToolRequest)
    ]

    checks: list[bool] = []
    notes: list[str] = []

    for entry in required:
        requirement: ToolUsageRequirement = (
            ToolUsageRequirement.of(entry) if isinstance(entry, str) else entry
        )
        satisfied: bool = any(_matches(request, requirement) for request in tool_requests)
        checks.append(satisfied)
        if not satisfied:
            notes.append(
                f"Required tool '{requirement.tool}' was not invoked"
                + (" with the expected arguments." if requirement.arguments else ".")
            )

    if expected:
        expected_set: set[str] = set(expected)
        expected_satisfied: bool = any(request.tool in expected_set for request in tool_requests)
        checks.append(expected_satisfied)
        if not expected_satisfied:
            notes.append("None of the expected tools were invoked: " + ", ".join(expected) + ".")

    if forbidden:
        forbidden_set: set[str] = set(forbidden)
        used_forbidden: list[str] = sorted(
            {request.tool for request in tool_requests if request.tool in forbidden_set}
        )
        checks.append(not used_forbidden)
        if used_forbidden:
            notes.append("Forbidden tool(s) invoked: " + ", ".join(used_forbidden) + ".")

    passed: int = sum(1 for check in checks if check)
    total: int = len(checks)
    value: float = (1.0 if passed == total else 0.0) if strict else passed / total

    if notes:
        return EvaluationScore.of(
            value,
            meta={"comment": " ".join(notes)},
        )

    return EvaluationScore.of(value)


def _matches(
    request: ModelToolRequest,
    requirement: ToolUsageRequirement,
    /,
) -> bool:
    if request.tool != requirement.tool:
        return False

    if requirement.arguments is None:
        return True

    return all(
        key in request.arguments and request.arguments[key] == value
        for key, value in requirement.arguments.items()
    )
