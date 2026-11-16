from pathlib import Path

import pytest
from haiway import ctx

from draive.skills import Skill


@pytest.mark.asyncio
async def test_skill_from_directory_accepts_path_below_a_symlinked_directory(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    skill_root = real / "my-skill"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text("---\nname: my-skill\ndescription: d\n---\nbody\n")
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)

    except OSError:
        pytest.skip("symlinks are not supported")

    async with ctx.scope("test"):
        skill = await Skill.from_directory(link / "my-skill")

    assert skill.name == "my-skill"
    assert skill.instructions == "body"
