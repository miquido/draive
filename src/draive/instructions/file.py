import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Formatter
from typing import Any, final

from haiway import asynchronous

from draive.commons import Meta
from draive.instructions.types import (
    Instruction,
    InstructionDeclaration,
    InstructionDeclarationArgument,
    InstructionMissing,
)

__all__ = ("InstructionsFileStorage",)


@final
class InstructionsFileStorage:
    def __init__(
        self,
        path: Path | str,
    ) -> None:
        self._path: Path
        match path:
            case str() as path_str:
                self._path = Path(path_str)

            case path:
                self._path = path

        self._storage: Mapping[str, str] | None = None
        self._listing: Sequence[InstructionDeclaration] | None = None

    async def listing(
        self,
        **extra: Any,
    ) -> Sequence[InstructionDeclaration]:
        if self._listing is not None:
            return self._listing

        if self._storage is None:
            self._storage = await self._file_load()

        formatter = Formatter()
        self._listing = tuple(
            InstructionDeclaration(
                name=name,
                arguments=tuple(
                    InstructionDeclarationArgument(name=arg_name)
                    for _, arg_name, _, _ in formatter.parse(content)
                    if arg_name  # we could also check for positional arguments
                ),
                meta=Meta({"file": str(self._path)}),
            )
            for name, content in self._storage.items()
        )

        return self._listing

    async def instruction(
        self,
        name: str,
        *,
        arguments: Mapping[str, str | float | int] | None = None,
        **extra: Any,
    ) -> Instruction:
        if self._storage is None:
            self._storage = await self._file_load()

        if instruction := self._storage.get(name):
            return Instruction.of(
                instruction,
                name=name,
                meta={"file": str(self._path)},
                arguments=arguments,
            )

        else:
            raise InstructionMissing(f"{self._path} does not contain instruction '{name}'")

    @asynchronous
    def _file_load(
        self,
    ) -> Mapping[str, str]:
        if self._path.exists():
            with open(self._path, mode="rb") as file:
                match json.loads(file.read()):
                    case {**elements}:
                        return {
                            key: value for key, value in elements.items() if isinstance(value, str)
                        }

                    case _:
                        return {}

        else:
            return {}
