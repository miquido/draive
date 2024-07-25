from pathlib import Path
from uuid import UUID

from draive.instructions.errors import MissingInstruction
from draive.instructions.types import Instruction, InstructionFetching
from draive.parameters import DataModel
from draive.utils import asynchronous

__all__ = [
    "instructions_file",
]


def instructions_file(
    path: Path | str,
) -> InstructionFetching:
    repository: _InstructionsFileStorage = _InstructionsFileStorage(path=path)

    async def fetch(
        key: str,
    ) -> Instruction:
        return await repository.instruction(key)

    return fetch


class _Instruction(DataModel):
    identifier: UUID
    content: str


class _Storage(DataModel):
    instructions: dict[str, _Instruction]


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

        self._storage: _Storage | None = None

    async def instruction(
        self,
        key: str,
    ) -> Instruction:
        if instruction := (await self.storage).instructions.get(key):
            return Instruction(
                instruction.content,
                identifier=instruction.identifier,
            )
        else:
            raise MissingInstruction(
                "%s does not contain instruction for key %s",
                self._path,
                key,
            )

    @property
    async def storage(self) -> _Storage:
        if cache := self._storage:
            return cache

        else:
            loaded: _Storage = await self.load()
            self._storage = loaded
            return loaded

    async def load(
        self,
    ) -> _Storage:
        return await self._file_load()

    @asynchronous(executor=None)
    def _file_load(
        self,
    ) -> _Storage:
        if self._path.exists():
            with open(self._path, mode="rb") as file:
                return _Storage.from_json(file.read())

        else:
            return _Storage(instructions={})
