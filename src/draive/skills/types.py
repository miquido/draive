import mimetypes
import re
from collections.abc import Iterator, Mapping, MutableSequence, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Self, final

import yaml
from haiway import (
    Alias,
    Directory,
    FileException,
    Files,
    Map,
    Meta,
    MetaValues,
    State,
    ValidationError,
    Verifier,
)

from draive.models import ModelInstructions
from draive.resources import ResourceContent
from draive.tools.function import tool
from draive.tools.types import Tool

__all__ = (
    "Skill",
    "SkillException",
    "SkillLoadingFailed",
    "SkillResource",
    "SkillResourceMissing",
)


class SkillException(Exception):
    """Base exception for skill-related failures.

    Parameters
    ----------
    *args : object
        Exception message arguments.
    skill : str
        Skill identifier associated with the failure.
    """

    __slots__ = ("skill",)

    def __init__(
        self,
        *args: object,
        skill: str,
    ) -> None:
        super().__init__(*args)
        self.skill: str = skill


@final
class SkillLoadingFailed(SkillException):
    """Raised when a skill directory cannot be accessed.

    Parameters
    ----------
    source : Path
        Skill directory that could not be loaded.
    """

    __slots__ = ("source",)

    def __init__(
        self,
        source: Path,
        /,
    ) -> None:
        super().__init__(
            f"Failed to load skill directory: {source}",
            skill=source.name or source.absolute().name,
        )
        self.source: Path = source


@final
class SkillResourceMissing(SkillException):
    """Raised when a requested skill resource cannot be resolved.

    Parameters
    ----------
    skill : str
        Skill identifier.
    path : str
        Requested relative resource path.
    """

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
        """Create a skill resource with normalized path and metadata.

        Parameters
        ----------
        path : str
            Relative resource path in POSIX format.
        content : ResourceContent
            Resource payload.
        meta : Meta | MetaValues | None, optional
            Additional resource metadata.

        Returns
        -------
        resource : Self
            Skill resource instance.

        Raises
        ------
        ValueError
            If `path` is invalid, absolute, or traverses outside skill root.
        """
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
    """Serializable definition of a Draive skill.

    Parameters
    ----------
    name : str
        Skill name.
    description : str
        Human-readable skill description.
    instructions : ModelInstructions
        Instruction text loaded from `SKILL.md`.
    resources : Mapping[str, SkillResource]
        Skill resources indexed by normalized relative path.
    meta : Meta, optional
        Additional skill metadata.
    """

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
        """Create a skill from explicit components.

        Parameters
        ----------
        name : str
            Skill name.
        description : str
            Human-readable skill description.
        instructions : str
            Instruction text.
        resources : Sequence[SkillResource], optional
            Resources to index by normalized path.
        meta : Meta | MetaValues | None, optional
            Additional skill metadata.

        Returns
        -------
        skill : Self
            Skill instance.
        """
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
        SkillLoadingFailed
            If the directory or one of its files cannot be accessed.
        ValueError
            If `SKILL.md` is missing or its contents are invalid.
        """
        root_path: Path = Path(path)
        skill_file_path: Path = root_path / "SKILL.md"
        try:
            root_entries = await Files.traverse(root_path)
            root_files: tuple[Path, ...] = tuple(
                entry.path for entry in root_entries if not isinstance(entry, Directory)
            )
            if skill_file_path not in root_files:
                raise ValueError(f"Missing SKILL.md in skill directory: {root_path}")

            parsed_skill: ParsedSkillFile
            async with Files.access(skill_file_path) as file:
                parsed_skill = ParsedSkillFile.from_file((await file.read()).decode("utf-8"))

            directory_name: str = root_path.name or root_path.absolute().name
            if parsed_skill.frontmatter.name != directory_name:
                raise ValueError(
                    "SKILL.md frontmatter name must match its parent directory: "
                    f"{parsed_skill.frontmatter.name} != {directory_name}"
                )

            resources: MutableSequence[SkillResource] = []
            for entry in await Files.traverse(root_path, recursive=True):
                if isinstance(entry, Directory):
                    continue

                file_path: Path = entry.path
                async with Files.access(file_path) as file:
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

        except FileException as exc:
            raise SkillLoadingFailed(root_path) from exc

        return cls.of(
            name=parsed_skill.frontmatter.name,
            description=parsed_skill.frontmatter.description,
            instructions=parsed_skill.instructions,
            resources=tuple(resources),
            meta=Meta.of(
                {
                    **parsed_skill.frontmatter.metadata,
                    **parsed_skill.frontmatter.optional_metadata,
                    "skill_source": str(root_path),
                }
            ),
        )

    name: str
    description: str
    instructions: ModelInstructions
    resources: Mapping[str, SkillResource] = Map()
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

    def resources_tool(
        self,
        name: str = "read_resource",
        description: str = "Read a resource file by relative path. "
        "Use this when you need access to instructions details or referenced resources.",
    ) -> Tool:

        @tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "format": "path",
                        "description": "Relative resource path",
                    },
                },
                "required": ("path",),
                "additionalProperties": False,
            },
        )
        async def load_resource(
            path: str,
        ) -> str | ResourceContent:
            content: ResourceContent = self.resource(path).content
            if _is_textual_mime_type(content.mime_type):
                try:
                    return content.to_bytes().decode("utf-8")

                except UnicodeDecodeError:
                    pass

            return content

        return load_resource


_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _verified_name(value: str) -> str:
    if not (1 <= len(value) <= 64):  # noqa: PLR2004
        raise ValueError("SKILL.md frontmatter name must be 1-64 characters long")

    if not _NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "SKILL.md frontmatter name must contain only lowercase letters, numbers, "
            "and single hyphens between alphanumeric segments"
        )

    return value


def _verified_description(value: str) -> str:
    if not (1 <= len(value) <= 1024):  # noqa: PLR2004
        raise ValueError("SKILL.md frontmatter description must be 1-1024 characters long")

    return value


def _verified_compatibility(value: str | None) -> str | None:
    if value is not None and not (1 <= len(value) <= 500):  # noqa: PLR2004
        raise ValueError("SKILL.md frontmatter compatibility must be 1-500 characters long")

    return value


@final
class ParsedFrontmatter(State):
    name: Annotated[str, Verifier(_verified_name)]
    description: Annotated[str, Verifier(_verified_description)]
    license: str | None = None
    compatibility: Annotated[str | None, Verifier(_verified_compatibility)] = None
    metadata: Mapping[str, str] = Map()
    allowed_tools: Annotated[str | None, Alias("allowed-tools")] = None

    @property
    def optional_metadata(self) -> Meta:
        return Meta.of(
            {
                **({"license": self.license} if self.license is not None else {}),
                **({"compatibility": self.compatibility} if self.compatibility is not None else {}),
                **({"allowed-tools": self.allowed_tools} if self.allowed_tools is not None else {}),
            }
        )


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
            frontmatter = yaml.safe_load("\n".join(frontmatter_lines))

        except yaml.YAMLError as exc:
            raise ValueError("SKILL.md frontmatter is malformed") from exc

        try:
            parsed_frontmatter = ParsedFrontmatter.from_mapping(frontmatter)

        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError(f"SKILL.md frontmatter is malformed: {exc}") from exc

        return ParsedSkillFile(
            frontmatter=parsed_frontmatter,
            instructions="\n".join(iterator),
        )

    frontmatter: ParsedFrontmatter
    instructions: ModelInstructions


def _normalize_path(path: str) -> str:
    normalized_path: PurePosixPath = PurePosixPath(path)

    if not normalized_path.parts:
        raise ValueError(f"Invalid resource path: {path}")

    if normalized_path.parts[0] == "/" or normalized_path.parts[0].startswith("~"):
        raise ValueError(f"Invalid resource path: {path}")

    for segment in normalized_path.parts:
        if segment == "..":
            raise ValueError(f"Invalid resource path: {path}")

    return str(normalized_path)


def _is_textual_mime_type(mime_type: str) -> bool:
    normalized: str = mime_type.partition(";")[0].lower()
    return (
        normalized.startswith("text/")
        or normalized
        in {
            "application/javascript",
            "application/json",
            "application/sql",
            "application/xml",
            "application/yaml",
        }
        or normalized.endswith(("+json", "+xml"))
    )
