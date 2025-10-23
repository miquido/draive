from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Final, Literal

from haiway import Immutable, ObservabilityLevel, ctx

from draive.guardrails.safety.types import GuardrailsSafetyException
from draive.multimodal import MultimodalContent, MultimodalContentPart, TextContent

__all__ = ("regex_safety_sanitization",)


class _RegexRule(Immutable):
    identifier: str
    pattern: re.Pattern[str]
    reason: str
    action: Literal["block", "mask"]
    replacement: str
    validator: Callable[[re.Match[str], str], bool] | None = None


_NEGATING_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:do|does|did|should|must|may|might|could|would|will|can)\s+not\b"
    r"|\b(?:don't|doesn't|didn't|shouldn't|mustn't|mayn't|mightn't|couldn't|"
    r"wouldn't|won't|can't|never)\b"
)
_CONDITIONAL_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:if|when|whenever|should|unless|in case)\s+you\b"
)
_COMMAND_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:please|kindly|now|just|simply|"
    r"you\s+(?:can|may|must|should|need|will|shall|are\s+allowed\s+to)|"
    r"i\s+(?:want|need)\s+you\s+to|"
    r"let(?:'s)?|from\s+now\s+on|for\s+the\s+rest\s+of\s+this\s+conversation|"
    r"act\s+as|pretend\s+to|assume\s+the\s+role)\b"
)


def _is_actionable_command(
    match: re.Match[str],
    text: str,
) -> bool:
    prefix: str = text[max(0, match.start() - 80) : match.start()]
    for separator in (".", "!", "?", "\n"):
        if separator in prefix:
            prefix = prefix.split(separator)[-1]

    clause: str = prefix.strip()
    if not clause:
        return True

    if _NEGATING_PREFIX.search(clause):
        return False

    if _CONDITIONAL_PREFIX.search(clause):
        return False

    return bool(_COMMAND_PREFIX.search(clause))


def _is_actionable_or_role_assignment(
    match: re.Match[str],
    text: str,
) -> bool:
    if _is_actionable_command(match, text):
        return True

    return bool(
        re.search(
            r"(?i)\byou\s+are\s+now\b|\byour\s+new\s+role\s+is\b|\bswitch\s+to\b",
            match.group(0),
        )
    )


_SENSITIVE_KEYWORDS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\binternal\b",
        r"\bhidden\b",
        r"\bsystem\b",
        r"\bsecret\b",
        r"\bexact\b",
        r"\bverbatim\b",
        r"\bconfidential\b",
        r"\bpolicy\b",
    )
)


def _contains_sensitive_language(
    match: re.Match[str],
    text: str,
) -> bool:
    window_start: int = max(0, match.start() - 40)
    window_end: int = min(len(text), match.end() + 40)
    window: str = text[window_start:window_end]
    return any(pattern.search(window) for pattern in _SENSITIVE_KEYWORDS)


def _requires_sensitive_context(
    match: re.Match[str],
    text: str,
) -> bool:
    # This validator previously inspected nearby punctuation to infer intent; it now
    # simply defers to the sensitive-language check so the behavior is explicit.
    return _contains_sensitive_language(match, text)


_RULES: Final[tuple[_RegexRule, ...]] = (
    _RegexRule(
        identifier="ignore_system_instructions",
        pattern=re.compile(
            r"(?i)\b(?:ignore|disregard|forget)\b"
            r".{0,80}\b(?:previous|prior|system)\b"
            r".{0,40}\b(?:instructions?|directives?|rules?)\b"
        ),
        reason="Detected attempt to override or ignore governing instructions.",
        action="block",
        replacement="[REDACTED:IGNORE]",
        validator=_is_actionable_command,
    ),
    _RegexRule(
        identifier="exfiltrate_system_prompt",
        pattern=re.compile(
            r"(?i)\b(?:reveal|show|expose|print|leak)\b"
            r".{0,80}\b(?:system|hidden|initial)\b"
            r".{0,40}\b(?:prompt|message|instructions?)\b"
        ),
        reason="Detected attempt to exfiltrate protected system prompt content.",
        action="block",
        replacement="[REDACTED:EXFILTRATION]",
        validator=_is_actionable_command,
    ),
    _RegexRule(
        identifier="disable_safety_mechanisms",
        pattern=re.compile(
            r"(?i)\b(?:disable|bypass|circumvent|override)\b"
            r".{0,80}\b(?:guardrails|safety|filters?|policies|restrictions?)\b"
        ),
        reason="Detected request to disable safety mechanisms.",
        action="block",
        replacement="[REDACTED:SAFETY_BYPASS]",
        validator=_is_actionable_command,
    ),
    _RegexRule(
        identifier="jailbreak_mode_activation",
        pattern=re.compile(
            r"(?i)\b(?:begin|enter|activate|start)\b"
            r".{0,40}\b(?:jailbreak|do[-\s]?anything[-\s]?now|dan\s*mode?)\b"
        ),
        reason="Detected explicit jailbreak mode activation attempt.",
        action="block",
        replacement="[REDACTED:JAILBREAK]",
        validator=_is_actionable_command,
    ),
    _RegexRule(
        identifier="role_override_injection",
        pattern=re.compile(
            r"(?is)\b(?:from\s+now\s+on|for\s+the\s+rest\s+of|regardless\s+of|"
            r"you\s+are\s+now|you\s+now\s+are|your\s+new\s+role\s+is|switch\s+to)\b"
            r".{0,120}\b(?:you\s+will|you\s+must|act\s+as|pretend\s+to|assume\s+the\s+role|"
            r"role:|mode)\b"
            r".{0,120}\b(?:system|root|developer|hacker|unfiltered|uncensored|"
            r"do[-\s]?anything|jailbreak|assistant)\b"
        ),
        reason="Detected attempt to coerce the assistant into an unsafe persona.",
        action="mask",
        replacement="[REDACTED:ROLE_OVERRIDE]",
        validator=_is_actionable_or_role_assignment,
    ),
    _RegexRule(
        identifier="role_override_direct_assignment",
        pattern=re.compile(
            r"(?i)\byou\s+are\s+now\b"
            r".{0,80}\b(?:developer|hacker|uncensored|unfiltered|dan|do[-\s]?anything|"
            r"system|assistant|persona)\b"
        ),
        reason="Detected attempt to assign an unsafe persona directly.",
        action="mask",
        replacement="[REDACTED:ROLE_OVERRIDE]",
        validator=_is_actionable_or_role_assignment,
    ),
    _RegexRule(
        identifier="policy_negation",
        pattern=re.compile(
            r"(?i)\b(?:ignore|without)\b"
            r".{0,40}\b(?:policy|rule|restriction|guardrail)s?\b"
            r".{0,80}\b(?:respond|comply|answer)\b"
        ),
        reason="Detected instruction to respond without applicable safety policies.",
        action="mask",
        replacement="[REDACTED:POLICY_NEGATION]",
        validator=_is_actionable_command,
    ),
    _RegexRule(
        identifier="tooling_inventory_exfiltration",
        pattern=re.compile(
            r"(?i)\b(?:list|show|reveal|describe|enumerate|tell|provide|explain|what|which)\b"
            r".{0,80}\b(?:all\s+)?(?:tools?|functions?|apis?|capabilities|commands?|instructions?)\b"
            r".{0,60}\b(?:available|you\s+have|you\s+can\s+use|internal|hidden|system|secret)\b"
        ),
        reason="Detected attempt to enumerate internal tools or instructions.",
        action="block",
        replacement="[REDACTED:TOOLS]",
        validator=_requires_sensitive_context,
    ),
    _RegexRule(
        identifier="instruction_manifest_exfiltration",
        pattern=re.compile(
            r"(?i)\b(?:summarize|reveal|list|show|print|explain|detail)\b"
            r".{0,80}\b(?:all\s+)?(?:system|safety|guardrail|policy|instruction|rule)s?\b"
            r".{0,60}\b(?:provided|you\s+follow|you\s+have|governing|hidden|internal)\b"
        ),
        reason="Detected attempt to retrieve governing rules or instructions verbatim.",
        action="block",
        replacement="[REDACTED:INSTRUCTIONS]",
        validator=_requires_sensitive_context,
    ),
)


def _sanitize_text(
    text: str,
    *,
    content: MultimodalContent,
) -> tuple[str, tuple[str, ...]]:
    redactions: list[str] = []
    sanitized_text: str = text
    for rule in _RULES:
        if rule.action == "block":
            for match in rule.pattern.finditer(sanitized_text):
                if rule.validator and not rule.validator(match, sanitized_text):
                    ctx.log_debug(f"Guardrails rule skipped after validation: {rule.identifier}")
                    continue

                ctx.log_warning(f"Guardrails safety rule triggered: {rule.identifier}")
                ctx.record(
                    ObservabilityLevel.WARNING,
                    metric="guardrails.safety.regex.blocks",
                    value=1,
                    unit="count",
                    kind="counter",
                    attributes={
                        "guardrails.safety.method": "regex",
                        "guardrails.safety.rule": rule.identifier,
                    },
                )
                ctx.record(
                    ObservabilityLevel.WARNING,
                    metric="guardrails.safety.regex.invocations",
                    value=1,
                    unit="count",
                    kind="counter",
                    attributes={
                        "guardrails.safety.method": "regex",
                        "guardrails.safety.outcome": "block",
                        "guardrails.safety.rule": rule.identifier,
                    },
                )

                raise GuardrailsSafetyException(
                    f"Guardrails safety blocked content by rule `{rule.identifier}`.",
                    reason=rule.reason,
                    content=content,
                )

            continue

        mutated_local: bool = False
        sanitized_text_local: str = sanitized_text

        def _replacement(
            match: re.Match[str],
            *,
            current_rule: _RegexRule = rule,
            working_text: str = sanitized_text_local,
        ) -> str:
            nonlocal mutated_local
            if current_rule.validator and not current_rule.validator(match, working_text):
                ctx.log_debug(
                    f"Guardrails rule skipped after validation: {current_rule.identifier}"
                )
                return match.group(0)

            mutated_local = True
            if current_rule.identifier not in redactions:
                redactions.append(current_rule.identifier)

            return current_rule.replacement

        updated_text: str = rule.pattern.sub(_replacement, sanitized_text)
        if mutated_local:
            sanitized_text = updated_text

    return sanitized_text, tuple(redactions)


async def regex_safety_sanitization(
    content: MultimodalContent,
    /,
    **extra: Any,
) -> MultimodalContent:
    """
    Perform deterministic regex-based jailbreak mitigation on multimodal content.

    Parameters
    ----------
    content : MultimodalContent
        The multimodal content to inspect and sanitize.
    **extra : Any
        Additional keyword arguments accepted for interface compatibility. These
        values are ignored.

    Returns
    -------
    MultimodalContent
        The sanitized content. When no sanitization is performed the original
        instance is returned.

    Raises
    ------
    GuardrailsSafetyException
        Raised when a high-severity jailbreak rule is detected.
    """
    async with ctx.scope("guardrails.safety.regex"):
        sanitized_parts: list[MultimodalContentPart] = []
        mutated: bool = False
        sanitized_count: int = 0
        rule_counts: dict[str, int] = {}

        for part in content.parts:
            if not isinstance(part, TextContent):
                sanitized_parts.append(part)
                continue

            original_text: str = part.to_str()
            sanitized_text, redactions = _sanitize_text(
                original_text,
                content=content,
            )
            for rule_identifier in redactions:
                rule_counts[rule_identifier] = rule_counts.get(rule_identifier, 0) + 1

            if not redactions and sanitized_text == original_text:
                sanitized_parts.append(part)
                continue

            mutated = True
            sanitized_count += 1
            joined_rules: str = ", ".join(redactions)
            ctx.log_debug(f"Guardrails applied sanitization for rules: {joined_rules}")
            sanitized_parts.append(
                TextContent.of(
                    sanitized_text,
                    meta=part.meta,
                )
            )

        if not mutated:
            ctx.record(
                ObservabilityLevel.INFO,
                metric="guardrails.safety.regex.passes",
                value=1,
                unit="count",
                kind="counter",
                attributes={
                    "guardrails.safety.method": "regex",
                    "guardrails.safety.rules": (),
                },
            )
            ctx.record(
                ObservabilityLevel.INFO,
                metric="guardrails.safety.regex.invocations",
                value=1,
                unit="count",
                kind="counter",
                attributes={
                    "guardrails.safety.method": "regex",
                    "guardrails.safety.outcome": "pass",
                },
            )
            result: MultimodalContent = content

        else:
            ctx.record(
                ObservabilityLevel.INFO,
                metric="guardrails.safety.regex.sanitized_parts",
                value=sanitized_count,
                unit="parts",
                kind="histogram",
                attributes={
                    "guardrails.safety.method": "regex",
                    "guardrails.safety.rules": tuple(sorted(rule_counts)),
                },
            )
            for rule_identifier, count in rule_counts.items():
                ctx.record(
                    ObservabilityLevel.INFO,
                    metric="guardrails.safety.regex.sanitized_rules",
                    value=count,
                    unit="parts",
                    kind="counter",
                    attributes={
                        "guardrails.safety.method": "regex",
                        "guardrails.safety.rule": rule_identifier,
                    },
                )
            ctx.record(
                ObservabilityLevel.INFO,
                metric="guardrails.safety.regex.invocations",
                value=1,
                unit="count",
                kind="counter",
                attributes={
                    "guardrails.safety.method": "regex",
                    "guardrails.safety.outcome": "sanitize",
                    "guardrails.safety.parts": sanitized_count,
                },
            )
            result = MultimodalContent(parts=tuple(sanitized_parts))

    return result
