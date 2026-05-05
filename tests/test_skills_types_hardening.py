from pathlib import Path

import pytest

from draive import ctx
from draive.resources import ResourceContent
from draive.skills.types import Skill, SkillResource


@pytest.mark.asyncio
async def test_skill_from_directory_parses_frontmatter_block(tmp_path: Path) -> None:
    skill_root = tmp_path / "csv-analysis"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: csv-analysis\n"
        "description: Use this skill when analyzing csv files.\n"
        "---\n"
        "# CSV Analysis\n"
        "\n"
        "Do the work.\n",
        encoding="utf-8",
    )

    async with ctx.scope("test"):
        skill = await Skill.from_directory(skill_root)

    assert skill.name == "csv-analysis"
    assert skill.description == "Use this skill when analyzing csv files."
    assert "Do the work." in skill.instructions


@pytest.mark.asyncio
async def test_skill_from_directory_skips_symlinked_resources(tmp_path: Path) -> None:
    skill_root = tmp_path / "safe-skill"
    skill_root.mkdir(parents=True)

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    (skill_root / "SKILL.md").write_text(
        "---\n"
        "name: safe-skill\n"
        "description: Valid description.\n"
        "---\n"
        "content\n",
        encoding="utf-8",
    )

    (skill_root / "leak.txt").symlink_to(outside)

    async with ctx.scope("test"):
        skill = await Skill.from_directory(skill_root)

    assert not skill.has_resource("leak.txt")


def test_skill_resource_rejects_parent_segments_in_path() -> None:
    with pytest.raises(ValueError):
        SkillResource.of(
            "nested/../escape.txt",
            content=ResourceContent.of(b"data", mime_type="text/plain"),
        )
