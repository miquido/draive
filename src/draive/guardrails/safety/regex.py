import re
from collections.abc import Callable, MutableSequence, Sequence
from typing import Any, Final

from haiway import Immutable, ctx

from draive.guardrails.safety.types import GuardrailsSafetyException, GuardrailsSafetySanitization
from draive.multimodal import MultimodalContent, MultimodalContentPart, TextContent

__all__ = (
    "RegexSanitizationRule",
    "guardrails_regex_sanitizer",
)


class RegexSanitizationRule(Immutable):
    """
    Immutable regular-expression safety rule definition.

    Parameters
    ----------
    identifier : str
        Stable rule identifier used for logging and metrics.
    pattern : re.Pattern[str]
        Compiled regular expression that triggers the rule.
    reason : str
        Human-readable reason presented when the rule blocks content.
    mask : str
        Replacement text used when rule is triggered. The rule will block content when None.
    validator : Callable[[re.Match[str], str], bool] or None, optional
        Optional predicate invoked to confirm the match should lead to the rule
        action. When ``False`` the match is ignored.
    """

    identifier: str
    pattern: re.Pattern[str]
    reason: str
    mask: str | None
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
    normalized_prefix: str = re.sub(
        r"[:;]|(?:\s[-\u2013\u2014]\s?)",
        "\n",
        prefix,
    )
    for separator in (".", "!", "?", "\n"):
        if separator in normalized_prefix:
            normalized_prefix = normalized_prefix.split(separator)[-1]

    clause: str = normalized_prefix.strip()
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
_ASSISTANT_TARGETED_KEYWORDS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bavailable\b",
        r"\byou\s+can\s+use\b",
        r"\byou\s+have\b",
    )
)
_ASSISTANT_REFERENCE: Final[re.Pattern[str]] = re.compile(r"(?i)\b(?:you|your|yours|assistant)\b")


def _contains_sensitive_language(
    match: re.Match[str],
    text: str,
) -> bool:
    window_start: int = max(0, match.start() - 40)
    window_end: int = min(len(text), match.end() + 40)
    window: str = text[window_start:window_end]
    if any(pattern.search(window) for pattern in _SENSITIVE_KEYWORDS):
        return True

    if any(pattern.search(window) for pattern in _ASSISTANT_TARGETED_KEYWORDS):
        return bool(_ASSISTANT_REFERENCE.search(window))

    return False


DEFAULT_REGEX_RULES: Final[tuple[RegexSanitizationRule, ...]] = (
    RegexSanitizationRule(
        identifier="ignore_system_instructions",
        pattern=re.compile(
            r"(?i)\b(?:ignore|disregard|forget)\b"
            r".{0,80}\b(?:previous|prior|system)\b"
            r".{0,40}\b(?:instructions?|directives?|rules?)\b"
        ),
        reason="Detected attempt to override or ignore governing instructions.",
        mask=None,
        validator=_is_actionable_command,
    ),
    RegexSanitizationRule(
        identifier="exfiltrate_system_prompt",
        pattern=re.compile(
            r"(?i)\b(?:reveal|show|expose|print|leak)\b"
            r".{0,80}\b(?:system|hidden|initial)\b"
            r".{0,40}\b(?:prompt|message|instructions?)\b"
        ),
        reason="Detected attempt to exfiltrate protected system prompt content.",
        mask=None,
        validator=_is_actionable_command,
    ),
    RegexSanitizationRule(
        identifier="disable_safety_mechanisms",
        pattern=re.compile(
            r"(?i)\b(?:disable|bypass|circumvent|override)\b"
            r".{0,80}\b(?:guardrails|safety|filters?|policies|restrictions?)\b"
        ),
        reason="Detected request to disable safety mechanisms.",
        mask=None,
        validator=_is_actionable_command,
    ),
    RegexSanitizationRule(
        identifier="jailbreak_mode_activation",
        pattern=re.compile(
            r"(?i)\b(?:begin|enter|activate|start)\b"
            r".{0,40}\b(?:jailbreak|do[-\s]?anything[-\s]?now|dan\s*mode?)\b"
        ),
        reason="Detected explicit jailbreak mode activation attempt.",
        mask=None,
        validator=_is_actionable_command,
    ),
    RegexSanitizationRule(
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
        mask="[REDACTED:ROLE_OVERRIDE]",
        validator=_is_actionable_or_role_assignment,
    ),
    RegexSanitizationRule(
        identifier="role_override_direct_assignment",
        pattern=re.compile(
            r"(?i)\byou\s+are\s+now\b"
            r".{0,80}\b(?:developer|hacker|uncensored|unfiltered|dan|do[-\s]?anything|"
            r"system|assistant|persona)\b"
        ),
        reason="Detected attempt to assign an unsafe persona directly.",
        mask="[REDACTED:ROLE_OVERRIDE]",
        validator=_is_actionable_or_role_assignment,
    ),
    RegexSanitizationRule(
        identifier="policy_negation",
        pattern=re.compile(
            r"(?i)\b(?:ignore|without)\b"
            r".{0,40}\b(?:policy|rule|restriction|guardrail)s?\b"
            r".{0,80}\b(?:respond|comply|answer)\b"
        ),
        reason="Detected instruction to respond without applicable safety policies.",
        mask="[REDACTED:POLICY_NEGATION]",
        validator=_is_actionable_command,
    ),
    RegexSanitizationRule(
        identifier="tooling_inventory_exfiltration",
        pattern=re.compile(
            r"(?i)\b(?:list|show|reveal|describe|enumerate|tell|provide|explain|what|which)\b"
            r".{0,80}\b(?:all\s+)?(?:tools?|functions?|apis?|capabilities|commands?|instructions?)\b"
            r".{0,60}\b(?:available|you\s+have|you\s+can\s+use|internal|hidden|system|secret)\b"
        ),
        reason="Detected attempt to enumerate internal tools or instructions.",
        mask=None,
        validator=_contains_sensitive_language,
    ),
    RegexSanitizationRule(
        identifier="instruction_manifest_exfiltration",
        pattern=re.compile(
            r"(?i)\b(?:summarize|reveal|list|show|print|explain|detail)\b"
            r".{0,80}\b(?:all\s+)?(?:system|safety|guardrail|policy|instruction|rule)s?\b"
            r".{0,60}\b(?:provided|you\s+follow|you\s+have|governing|hidden|internal)\b"
        ),
        reason="Detected attempt to retrieve governing rules or instructions verbatim.",
        mask=None,
        validator=_contains_sensitive_language,
    ),
    RegexSanitizationRule(
        identifier="binary_encoded_instructions",
        pattern=re.compile(r"(?i)(?:\b[01]{8}\b(?:[\s,;:|/\\-]+\b[01]{8}\b){5,})"),
        reason="Detected attempt to convey instructions via binary encoding.",
        mask=None,
    ),
    RegexSanitizationRule(
        identifier="base64_encoded_instructions",
        pattern=re.compile(
            r"(?i)(?:\b(?:decode|interpret|base64)\b.{0,40})?"
            r"([A-Za-z0-9+/]{48,}={0,2})"
        ),
        reason="Detected attempt to convey instructions via base64 encoding.",
        mask=None,
    ),
    RegexSanitizationRule(
        identifier="hex_encoded_instructions",
        pattern=re.compile(
            r"(?i)(?:\b(?:decode|interpret|hex|hexadecimal)\b.{0,40})?"
            r"(?:(?:0x[0-9a-f]{2}\b|[0-9a-f]{2}\b)(?:[\s,;:|/\\-]+(?:0x[0-9a-f]{2}\b|[0-9a-f]{2}\b)){5,})"
        ),
        reason="Detected attempt to convey instructions via hexadecimal encoding.",
        mask=None,
    ),
)


def _sanitize_text(
    text: str,
    *,
    content: MultimodalContent,
    rules: Sequence[RegexSanitizationRule],
) -> str:
    sanitized_text: str = text
    for rule in rules:
        match rule.mask:
            case None:
                for match in rule.pattern.finditer(sanitized_text):
                    if rule.validator and not rule.validator(match, sanitized_text):
                        ctx.log_debug(
                            f"Guardrails rule skipped after validation: {rule.identifier}"
                        )
                        continue

                    ctx.log_warning(f"Guardrails safety rule triggered: {rule.identifier}")

                    raise GuardrailsSafetyException(
                        f"Guardrails safety blocked content by rule `{rule.identifier}`.",
                        reason=rule.reason,
                        content=content,
                    )

            case str() as mask:

                def _replacement(
                    match: re.Match[str],
                    *,
                    current_rule: RegexSanitizationRule = rule,
                    working_text: str = sanitized_text,
                ) -> str:
                    if current_rule.validator and not current_rule.validator(match, working_text):
                        ctx.log_debug(
                            f"Guardrails rule skipped after validation: {current_rule.identifier}"
                        )
                        return match.group(0)

                    ctx.log_warning(f"Guardrails safety rule triggered: {current_rule.identifier}")

                    return mask

                sanitized_text = rule.pattern.sub(_replacement, sanitized_text)

    return sanitized_text


async def _sanitize_content(
    content: MultimodalContent,
    *,
    rules: Sequence[RegexSanitizationRule],
) -> MultimodalContent:
    sanitized_parts: MutableSequence[MultimodalContentPart] = []
    mutated: bool = False

    for part in content.parts:
        if not isinstance(part, TextContent):
            sanitized_parts.append(part)
            continue  # pass non-text parts as is

        sanitized_text: str = _sanitize_text(
            part.text,
            content=content,
            rules=rules,
        )

        if sanitized_text != part.text:
            mutated = True
            sanitized_parts.append(
                TextContent(
                    text=sanitized_text,
                    meta=part.meta,
                )
            )

        else:
            sanitized_parts.append(part)

    if not mutated:
        return content

    return MultimodalContent(parts=tuple(sanitized_parts))


def guardrails_regex_sanitizer(
    *,
    rules: Sequence[RegexSanitizationRule] = DEFAULT_REGEX_RULES,
) -> GuardrailsSafetySanitization:
    """
    Build a regex-based guardrail sanitizer for a specific rule set.

    Parameters
    ----------
    rules : Sequence[RegexSanitizationRule]
        Rules to apply sequentially when inspecting text content.

    Returns
    -------
    GuardrailsSafetySanitization
        Sanitizer compatible with guardrail pipelines.
    """

    assert len(rules) == len({rule.identifier for rule in rules})  # nosec: B101

    async def _sanitizer(
        content: MultimodalContent,
        /,
        **extra: Any,
    ) -> MultimodalContent:
        async with ctx.scope("guardrails.safety.sanitize"):
            return await _sanitize_content(
                content,
                rules=rules,
            )

    return _sanitizer
