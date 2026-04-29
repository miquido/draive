from pathlib import Path

import pytest

from draive import ctx
from draive.resources import ResourceContent
from draive.skills import Skill, SkillResource, SkillResourceMissing


def test_skill_resource_lookup_uses_normalized_paths() -> None:
    skill = Skill.of(
        "csv-analysis",
        description="Use when user asks to analyze spreadsheets",
        instructions="Analyze CSV files using references.",
        resources=(
            SkillResource.of(
                "references/guide.md",
                content=ResourceContent.of(b"# guide", mime_type="text/plain"),
            ),
        ),
    )

    resolved = skill.resource("./references/guide.md")

    assert resolved.path == "references/guide.md"
    assert resolved.content.to_bytes() == b"# guide"


def test_skill_resource_lookup_raises_when_missing() -> None:
    skill = Skill.of("example", description="desc", instructions="Follow instructions")

    with pytest.raises(SkillResourceMissing) as exc_info:
        skill.resource("references/missing.md")

    assert exc_info.value.skill == "example"
    assert exc_info.value.path == "references/missing.md"


def test_skill_rejects_absolute_or_parent_resource_paths() -> None:
    with pytest.raises(ValueError):
        SkillResource.of(
            "/references/guide.md",
            content=ResourceContent.of(b"nope", mime_type="text/plain"),
        )

    with pytest.raises(ValueError):
        SkillResource.of(
            "../references/guide.md",
            content=ResourceContent.of(b"nope", mime_type="text/plain"),
        )


@pytest.mark.asyncio
async def test_skill_from_directory_loads_frontmatter_and_resources(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "csv-analysis"
    references_dir = skill_root / "references"
    assets_dir = skill_root / "assets"
    references_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)

    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: csv-analysis\n"
        "description: Use this skill when analyzing csv files.\n"
        "metadata:\n"
        "  author: example-org\n"
        '  version: "1.0"\n'
        "---\n"
        "# CSV Analysis\n"
        "\n"
        "## Quick Start\n"
        "- Read references/guide.md\n",
        encoding="utf-8",
    )
    (references_dir / "guide.md").write_text("# guide", encoding="utf-8")
    (assets_dir / "schema.json").write_text('{"ok":true}', encoding="utf-8")

    async with ctx.scope("test"):
        skill = await Skill.from_directory(skill_root)

    assert skill.name == "csv-analysis"
    assert skill.description == "Use this skill when analyzing csv files."
    assert skill.meta["author"] == "example-org"
    assert skill.meta["version"] == "1.0"
    assert skill.instructions is not None
    assert "Quick Start" in skill.instructions
    assert skill.has_resource("SKILL.md")
    assert skill.has_resource("references/guide.md")
    assert skill.has_resource("assets/schema.json")


@pytest.mark.asyncio
async def test_skill_from_directory_requires_frontmatter(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "missing-frontmatter"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text("# plain markdown", encoding="utf-8")

    async with ctx.scope("test"):
        with pytest.raises(ValueError):
            await Skill.from_directory(skill_root)


@pytest.mark.asyncio
async def test_skill_from_directory_supports_crlf_frontmatter(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "crlf-frontmatter"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\r\n"
        "name: crlf-skill\r\n"
        "description: Works with Windows newlines.\r\n"
        "---\r\n"
        "# Body\r\n",
        encoding="utf-8",
        newline="",
    )

    async with ctx.scope("test"):
        skill = await Skill.from_directory(skill_root)

    assert skill.name == "crlf-skill"
    assert skill.description == "Works with Windows newlines."


@pytest.mark.asyncio
async def test_skill_from_directory_raises_value_error_for_malformed_yaml(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "malformed-frontmatter"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: [\n"
        "description: invalid yaml\n"
        "---\n"
        "# Body\n",
        encoding="utf-8",
    )

    async with ctx.scope("test"):
        with pytest.raises(ValueError):
            await Skill.from_directory(skill_root)


@pytest.mark.asyncio
async def test_skill_from_directory_raises_value_error_for_empty_frontmatter(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "empty-frontmatter"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "---\n"
        "# Body\n",
        encoding="utf-8",
    )

    async with ctx.scope("test"):
        with pytest.raises(ValueError):
            await Skill.from_directory(skill_root)


@pytest.mark.asyncio
async def test_skill_from_directory_does_not_require_name_matching_directory(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "csv-analysis"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: other-name\n"
        "description: Use this skill when analyzing csv files.\n"
        "---\n"
        "# CSV Analysis\n",
        encoding="utf-8",
    )

    async with ctx.scope("test"):
        skill = await Skill.from_directory(skill_root)

    assert skill.name == "other-name"


@pytest.mark.asyncio
async def test_skill_from_directory_allows_invalid_name_format(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "csv-analysis"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: csv--analysis\n"
        "description: Use this skill when analyzing csv files.\n"
        "---\n"
        "# CSV Analysis\n",
        encoding="utf-8",
    )

    async with ctx.scope("test"):
        skill = await Skill.from_directory(skill_root)

    assert skill.name == "csv--analysis"


@pytest.mark.asyncio
async def test_skill_from_directory_allows_non_string_metadata_values(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / "csv-analysis"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: csv-analysis\n"
        "description: Use this skill when analyzing csv files.\n"
        "metadata:\n"
        "  version: 1\n"
        "---\n"
        "# CSV Analysis\n",
        encoding="utf-8",
    )

    async with ctx.scope("test"):
        skill = await Skill.from_directory(skill_root)

    assert skill.meta["version"] == 1
