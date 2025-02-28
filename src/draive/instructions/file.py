import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from haiway import asynchronous

from draive.instructions.types import Instruction, InstructionFetching, MissingInstruction

__all__ = [
    "instructions_file",
]


def instructions_file(
    path: Path | str,
) -> InstructionFetching:
    storage: _InstructionsFileStorage = _InstructionsFileStorage(path=path)

    async def fetch(
        identifier: str,
        /,
        *,
        arguments: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> Instruction:
        return await storage.instruction(
            identifier,
            arguments=arguments,
            **extra,
        )

    return fetch


class _InstructionsFileStorage:
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

    async def instruction(
        self,
        name: str,
        *,
        arguments: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> Instruction:
        if self._storage is None:
            self._storage = await self._file_load()

        if instruction := self._storage.get(name):
            return Instruction.of(
                instruction,
                name=name,
                meta={"file": str(self._path)},
                **(arguments if arguments is not None else {}),
            )

        else:
            raise MissingInstruction(
                "%s does not contain instruction for identifier %s",
                self._path,
                name,
            )

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
