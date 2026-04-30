import mimetypes
from collections.abc import Iterator, Mapping, MutableSequence, Sequence
from pathlib import Path, PurePosixPath
from typing import Self, final

import yaml
from haiway import FileAccess, Meta, MetaValues, State

from draive.models import ModelInstructions
from draive.resources import ResourceContent

__all__ = (
    "Skill",
    "SkillException",
    "SkillResource",
    "SkillResourceMissing",
)


class SkillException(Exception):
    __slots__ = ("skill",)

    def __init__(
        self,
        *args: object,
        skill: str,
    ) -> None:
        super().__init__(*args)
        self.skill: str = skill


@final
class SkillResourceMissing(SkillException):
    __slots__ = ("path",)

    def __init__(
        self,
        skill: str,
        *,
        path: str,
    ) -> None:
        super().__init__(
            f"Missing skill resource - {skill}:{path}",
            skill=skill,
        )
        self.path: str = path


@final
class SkillResource(State, serializable=True):
    """Single skill resource referenced by relative path.

    Parameters
    ----------
    path : str
        Relative resource path in POSIX format.
    content : ResourceContent
        Resource payload.
    meta : Meta | MetaValues | None, optional
        Additional resource metadata.
    """

    @classmethod
    def of(
        cls,
        path: str,
        *,
        content: ResourceContent,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            path=_normalize_path(path),
            content=content,
            meta=Meta.of(meta),
        )

    path: str
    content: ResourceContent
    meta: Meta = Meta.empty


@final
class Skill(State, serializable=True):
    @classmethod
    def of(
        cls,
        /,
        name: str,
        *,
        description: str,
        instructions: str,
        resources: Sequence[SkillResource] = (),
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            name=name,
            description=description,
            instructions=instructions,
            resources={resource.path: resource for resource in resources},
            meta=Meta.of(meta),
        )

    @classmethod
    async def from_directory(
        cls,
        path: Path | str,
        /,
    ) -> Self:
        """Load a skill from a directory with `SKILL.md` and bundled files.

        Parameters
        ----------
        path : Path | str
            Skill root directory path.

        Returns
        -------
        skill : Self
            Loaded skill with metadata parsed from `SKILL.md` frontmatter and
            resources indexed by relative paths.

        Raises
        ------
        ValueError
            If `path` is not a directory, if `SKILL.md` is missing, or when
            frontmatter is malformed.
        """
        root_path: Path = Path(path)
        if not root_path.is_dir():
            raise ValueError(f"Skill path is not a directory: {root_path}")

        skill_file_path: Path = root_path / "SKILL.md"
        if not skill_file_path.is_file():
            raise ValueError(f"Missing SKILL.md in skill directory: {root_path}")

        parsed_skill: ParsedSkillFile
        async with FileAccess.open(skill_file_path) as file:
            parsed_skill = ParsedSkillFile.from_file((await file.read()).decode("utf-8"))

        resources: MutableSequence[SkillResource] = []
        resolved_root_path: Path = root_path.resolve(strict=True)
        for file_path in root_path.rglob("*"):
            if not file_path.is_file() or file_path.is_symlink():
                continue  # skip symlinks and directories

            resolved_file_path: Path = file_path.resolve(strict=True)
            if resolved_root_path not in resolved_file_path.parents:
                raise ValueError(f"Skill resource points outside skill directory: {file_path}")

            async with FileAccess.open(file_path) as file:
                mime_type: str | None
                mime_type, _ = mimetypes.guess_type(file_path.name)
                resources.append(
                    SkillResource.of(
                        file_path.relative_to(root_path).as_posix(),
                        content=ResourceContent.of(
                            await file.read(),
                            mime_type=mime_type or "application/octet-stream",
                        ),
                    )
                )

        return cls.of(
            name=parsed_skill.frontmatter.name,
            description=parsed_skill.frontmatter.description,
            instructions=parsed_skill.instructions,
            resources=tuple(resources),
            meta=Meta.of({"source": str(root_path), **parsed_skill.frontmatter.metadata}),
        )

    name: str
    description: str
    instructions: ModelInstructions
    resources: Mapping[str, SkillResource]
    meta: Meta = Meta.empty

    def resource(
        self,
        path: str,
        /,
    ) -> SkillResource:
        """Read resource by a relative path-like identifier.

        Parameters
        ----------
        path : str
            Relative POSIX-like path. Supports `./` path segments.

        Returns
        -------
        resource : SkillResource
            Matched skill resource.

        Raises
        ------
        SkillResourceMissing
            When no resource exists under the normalized path.
        ValueError
            When path is absolute or traverses outside the skill root.
        """
        if resolved := self.resources.get(_normalize_path(path)):
            return resolved

        raise SkillResourceMissing(
            self.name,
            path=path,
        )

    def has_resource(
        self,
        path: str,
        /,
    ) -> bool:
        """Check if a resource exists under a relative path.

        Parameters
        ----------
        path : str
            Relative POSIX-like path.

        Returns
        -------
        exists : bool
            `True` when resource can be resolved.
        """
        return _normalize_path(path) in self.resources


@final
class ParsedFrontmatter(State):
    name: str
    description: str
    metadata: Meta = Meta.empty


@final
class ParsedSkillFile(State):
    @classmethod
    def from_file(
        cls,
        content: str,
    ) -> ParsedSkillFile:
        if not content:
            raise ValueError("SKILL.md is empty")

        frontmatter_lines: MutableSequence[str] = []
        lines: Sequence[str] = content.splitlines()
        iterator: Iterator[str] = iter(lines)
        line: str | None = next(iterator, None)
        if line != "---":
            raise ValueError("SKILL.md frontmatter is missing")

        line = next(iterator, None)
        while line is not None:
            if line == "---":
                break

            else:
                frontmatter_lines.append(line)
                line = next(iterator, None)

        else:
            raise ValueError("SKILL.md frontmatter is not closed")

        try:
            return ParsedSkillFile(
                frontmatter=ParsedFrontmatter.from_mapping(
                    yaml.safe_load("\n".join(frontmatter_lines))
                ),
                instructions="\n".join(iterator),
            )

        except Exception as exc:
            raise ValueError("SKILL.md frontmatter is malformed") from exc

    frontmatter: ParsedFrontmatter
    instructions: ModelInstructions


def _normalize_path(path: str) -> str:
    normalized_path: PurePosixPath = PurePosixPath(path)

    if not normalized_path.parts:
        raise ValueError(f"Invalid resource path: {path}")

    if normalized_path.parts[0] in {"/", "~"}:
        raise ValueError(f"Invalid resource path: {path}")

    for segment in normalized_path.parts:
        if segment == "..":
            raise ValueError(f"Invalid resource path: {path}")

    return str(normalized_path)
