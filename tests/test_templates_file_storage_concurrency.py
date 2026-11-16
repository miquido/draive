import asyncio
import json
from pathlib import Path

import pytest
from haiway import ctx

from draive.multimodal import Template, TemplateDeclaration, TemplatesRepository


def _declaration(identifier: str) -> TemplateDeclaration:
    return TemplateDeclaration.of(identifier, variables={"name": "name"})


@pytest.mark.asyncio
async def test_file_storage_persists_and_reloads_templates(tmp_path: Path) -> None:
    path: Path = tmp_path.resolve() / "templates.json"

    async with ctx.scope("write", TemplatesRepository.file(path)):
        await TemplatesRepository.define(_declaration("hello"), content="日本 {%name%}")

    async with ctx.scope("read", TemplatesRepository.file(path)):
        resolved = await TemplatesRepository.resolve_str(
            Template.of("hello"),
            arguments={"name": "é"},
        )

    assert resolved == "日本 é"


@pytest.mark.asyncio
async def test_file_storage_sequential_defines_are_all_kept(tmp_path: Path) -> None:
    path: Path = tmp_path.resolve() / "templates.json"

    async with ctx.scope("write", TemplatesRepository.file(path)):
        for index in range(10):
            await TemplatesRepository.define(_declaration(f"t{index}"), content="{%name%}")

    assert len(json.loads(path.read_text())) == 10


@pytest.mark.asyncio
async def test_file_storage_refuses_to_overwrite_invalid_file(tmp_path: Path) -> None:
    path: Path = tmp_path.resolve() / "templates.json"
    path.write_text("{not json")

    async with ctx.scope("write", TemplatesRepository.file(path)):
        with pytest.raises(Exception, match="refusing to overwrite"):
            await TemplatesRepository.define(_declaration("hello"), content="{%name%}")

    assert path.read_text() == "{not json"


@pytest.mark.asyncio
async def test_file_storage_concurrent_first_defines_keep_all_templates(tmp_path: Path) -> None:
    path: Path = tmp_path.resolve() / "templates.json"

    async with ctx.scope("write", TemplatesRepository.file(path)):
        await asyncio.gather(
            *(
                TemplatesRepository.define(_declaration(f"t{index}"), content="{%name%}")
                for index in range(20)
            )
        )
        listed = await TemplatesRepository.templates()

    assert len(listed.items) == 20
    assert len(json.loads(path.read_text())) == 20


@pytest.mark.asyncio
async def test_file_storage_concurrent_defines_keep_file_valid(tmp_path: Path) -> None:
    path: Path = tmp_path.resolve() / "templates.json"

    async with ctx.scope("write", TemplatesRepository.file(path)):
        await TemplatesRepository.define(_declaration("first"), content="{%name%}")
        await asyncio.gather(
            *(
                TemplatesRepository.define(_declaration(f"t{index}"), content="{%name%}")
                for index in range(20)
            )
        )

    assert len(json.loads(path.read_text())) == 21


@pytest.mark.asyncio
async def test_file_storage_failed_save_keeps_memory_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path: Path = tmp_path.resolve() / "templates.json"

    async with ctx.scope("write", TemplatesRepository.file(path)):
        await TemplatesRepository.define(_declaration("first"), content="{%name%}")

        def failing(*_: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(
            "draive.multimodal.templates.repository._write_atomically",
            failing,
        )
        with pytest.raises(OSError):
            await TemplatesRepository.define(_declaration("second"), content="{%name%}")

        listed = await TemplatesRepository.templates()

    assert [item.identifier for item in listed.items] == ["first"]
    assert len(json.loads(path.read_text())) == 1


@pytest.mark.asyncio
async def test_file_storage_fifty_concurrent_defines_with_repeats_stay_consistent(
    tmp_path: Path,
) -> None:
    path: Path = tmp_path.resolve() / "templates.json"

    async with ctx.scope("write", TemplatesRepository.file(path)):
        await asyncio.gather(
            *(
                TemplatesRepository.define(
                    _declaration(f"t{index % 40}"), content=f"v{index} {{%name%}}"
                )
                for index in range(50)
            )
        )
        listed = await TemplatesRepository.templates()

    stored = json.loads(path.read_text())
    assert len(listed.items) == 32  # default page size
    assert len(stored) == 40
