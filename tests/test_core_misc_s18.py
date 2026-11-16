from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import pytest
from haiway import State, ctx

from draive import (
    AgentMemory,
    ModelGeneration,
    MultimodalContent,
    Toolbox,
    tool,
)
from draive.generation import GenerationOutputMissing
from draive.models import model_rate_limit
from draive.skills.types import ParsedSkillFile, Skill
from draive.steps.decoding import decode_content


@tool
async def _noop() -> str:
    return "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image", "audio"])
async def test_media_generation_without_output_raises_typed_exception(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from draive.generation.audio.default import generate_audio
    from draive.generation.image.default import generate_image
    from draive.steps import Step

    class _Run:
        async def run(self, *_a: Any, **_k: Any) -> MultimodalContent:
            return MultimodalContent.of("no media")

    monkeypatch.setattr(Step, "generating_completion", staticmethod(lambda **_k: _Run()))
    generate = generate_image if kind == "image" else generate_audio
    async with ctx.scope("test"):
        with pytest.raises(GenerationOutputMissing):
            await generate(instructions="", input="make it")


@pytest.mark.asyncio
async def test_rate_limit_retry_after_http_date() -> None:
    future = datetime.now(UTC) + timedelta(seconds=120)
    async with ctx.scope("test"):
        limit = model_rate_limit(
            provider="p", model="m", retry_after=format_datetime(future, usegmt=True)
        )
    assert 100 <= limit.retry_after <= 121


@pytest.mark.asyncio
async def test_rate_limit_retry_after_past_http_date_clamped_to_zero() -> None:
    async with ctx.scope("test"):
        limit = model_rate_limit(
            provider="p", model="m", retry_after="Wed, 21 Oct 2015 07:28:00 GMT"
        )
    assert limit.retry_after == 0.0


@pytest.mark.asyncio
async def test_rate_limit_retry_after_naive_date_is_treated_as_utc() -> None:
    async with ctx.scope("test"):
        limit = model_rate_limit(provider="p", model="m", retry_after="21 Oct 2015 07:28:00 -0000")
    assert limit.retry_after == 0.0


_SKILL = "name: demo-skill\ndescription: Does things.\n"


def test_skill_frontmatter_tolerates_leading_bom() -> None:
    parsed = ParsedSkillFile.from_file(f"﻿---\n{_SKILL}---\nbody\n")
    assert parsed.frontmatter.name == "demo-skill"


def test_skill_frontmatter_unknown_keys_are_ignored() -> None:
    parsed = ParsedSkillFile.from_file(f"---\n{_SKILL}version: 2\ntags: [a, b]\n---\nbody\n")
    assert parsed.frontmatter.name == "demo-skill"
    assert parsed.instructions == "body"


@pytest.mark.asyncio
async def test_skill_from_directory_with_bom_and_extra_keys(tmp_path: Path) -> None:
    root = tmp_path / "demo-skill"
    root.mkdir()
    (root / "SKILL.md").write_text(f"﻿---\n{_SKILL}argument-hint: x\n---\nbody\n", encoding="utf-8")
    async with ctx.scope("test"):
        skill = await Skill.from_directory(root)
    assert skill.name == "demo-skill"


def test_toolbox_of_existing_toolbox_with_extra_tools_raises_value_error() -> None:
    with pytest.raises(ValueError, match="tools"):
        Toolbox.of(Toolbox.of(_noop), _noop)


def test_toolbox_of_existing_toolbox_with_suggesting_raises_value_error() -> None:
    with pytest.raises(ValueError, match="suggesting"):
        Toolbox.of(Toolbox.of(_noop), suggesting=True)


def test_toolbox_filtered_tools_and_tags_raises_value_error() -> None:
    with pytest.raises(ValueError, match="tags"):
        Toolbox.of(_noop).filtered(tools=["_noop"], tags=frozenset({"x"}))


@pytest.mark.parametrize("limit", [0, -1])
def test_agent_memory_volatile_rejects_non_positive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="threads_limit"):
        AgentMemory.volatile(threads_limit=limit)


class _NotSerializable:
    pass


@pytest.mark.asyncio
async def test_model_generation_rejects_non_serializable_type() -> None:
    class Plain(State):
        value: _NotSerializable

    async with ctx.scope("test"):
        with pytest.raises(TypeError, match="serializable"):
            await ModelGeneration.generate(Plain, input="x")  # type: ignore[arg-type]


class _Example(State):
    name: str


@pytest.mark.parametrize(
    "payload",
    ['{"name": "Paris", "extra": 1}', 'Here you go: {"name": "Paris"}'],
)
def test_default_decoding_is_strict_by_design(payload: str) -> None:
    # extra keys and prose around json are rejected; lenient reads use the `decoder=` hook
    with pytest.raises(Exception):  # noqa: B017
        decode_content(MultimodalContent.of(payload), _Example)


def test_custom_decoder_hook_allows_lenient_decoding() -> None:
    decoded = decode_content(
        MultimodalContent.of('Here you go: {"name": "Paris", "extra": 1}'),
        _Example,
        decoder=lambda content: _Example(name="Paris"),
    )
    assert decoded.name == "Paris"
