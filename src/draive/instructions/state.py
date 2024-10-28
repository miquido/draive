from collections.abc import Callable
from time import monotonic
from typing import NamedTuple

from haiway import State

from draive.instructions.errors import MissingInstruction
from draive.instructions.types import Instruction, InstructionFetching

__all__ = [
    "InstructionsRepository",
]


async def _missing(
    key: str,
) -> Instruction:
    raise MissingInstruction("InstructionsRepository is not defined!")


class _CacheEntry(NamedTuple):
    instruction: Instruction
    timestamp: float


_instructions_cache: dict[str, _CacheEntry] = {}


class InstructionsRepository(State):
    fetch: InstructionFetching = _missing
    cache_expiration: float | None = None
    clock: Callable[[], float] = monotonic

    async def instruction(
        self,
        key: str,
        /,
        *,
        default: Instruction | str | None = None,
    ) -> Instruction:
        try:
            if (cached := _instructions_cache.get(key, None)) and (
                self.cache_expiration is None
                or cached.timestamp + self.cache_expiration >= self.clock()
            ):
                return cached.instruction

            else:
                instruction: Instruction = await self.fetch(key)
                _instructions_cache[key] = _CacheEntry(
                    instruction=instruction,
                    timestamp=self.clock(),
                )
                return instruction

        except MissingInstruction as exc:
            match default:
                case None:
                    raise exc

                case str() as string:
                    instruction: Instruction = Instruction.of(string)
                    _instructions_cache[key] = _CacheEntry(
                        instruction=instruction,
                        timestamp=self.clock(),
                    )
                    return instruction

                case instruction:
                    _instructions_cache[key] = _CacheEntry(
                        instruction=instruction,
                        timestamp=self.clock(),
                    )
                    return instruction
