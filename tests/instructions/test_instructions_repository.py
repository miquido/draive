import pytest

from draive import Instruction, InstructionsRepository


class _Fetcher:
    def __init__(self, calls_limit: int):
        self.calls_counter: int = 0
        self.calls_limit: int = calls_limit

    async def fetcher(self, key: str) -> Instruction:
        self.calls_counter += 1

        if self.calls_counter > self.calls_limit:
            raise Exception(
                f"Fetch has been called too many times, expected at most {self.calls_limit}"
            )

        return Instruction.of(f"My instruction #{self.calls_counter} for {key=}")


class _Clock:
    def __init__(self):
        self.value: float = 0.0

    def ticker(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


@pytest.mark.asyncio
async def test_cache_never_expires() -> None:
    def _cache_key(key: str) -> str:
        return f"{test_cache_never_expires.__name__}::{key}"

    # given
    repository = InstructionsRepository(
        fetch=_Fetcher(2).fetcher,
    )

    # then
    _assert_instruction_contents(
        await repository.instruction(_cache_key("foo")),
        index=1,
        key=_cache_key("foo"),
    )

    _assert_instruction_contents(
        await repository.instruction(_cache_key("bar")),
        index=2,
        key=_cache_key("bar"),
    )

    _assert_instruction_contents(
        await repository.instruction(_cache_key("foo")),
        index=1,
        key=_cache_key("foo"),
    )

    _assert_instruction_contents(
        await repository.instruction(_cache_key("bar")),
        index=2,
        key=_cache_key("bar"),
    )


@pytest.mark.asyncio
async def test_cache_immediately_expires() -> None:
    def _cache_key(key: str) -> str:
        return f"{test_cache_immediately_expires.__name__}::{key}"

    # given
    clock = _Clock()
    repository = InstructionsRepository(
        fetch=_Fetcher(4).fetcher,
        clock=clock.ticker,
        cache_expiration=0.0,
    )

    # then
    _assert_instruction_contents(
        await repository.instruction(_cache_key("foo")),
        index=1,
        key=_cache_key("foo"),
    )

    _assert_instruction_contents(
        await repository.instruction(_cache_key("bar")),
        index=2,
        key=_cache_key("bar"),
    )

    # when: simulate any operation
    clock.advance(0.000_001)

    # then
    _assert_instruction_contents(
        await repository.instruction(_cache_key("foo")),
        index=3,
        key=_cache_key("foo"),
    )

    _assert_instruction_contents(
        await repository.instruction(_cache_key("bar")),
        index=4,
        key=_cache_key("bar"),
    )


@pytest.mark.asyncio
async def test_new_element_is_created_when_cache_expires() -> None:
    def _cache_key(key: str) -> str:
        return f"{test_new_element_is_created_when_cache_expires.__name__}::{key}"

    # given
    clock = _Clock()
    repository = InstructionsRepository(
        fetch=_Fetcher(5).fetcher,
        clock=clock.ticker,
        cache_expiration=10.0,
    )

    # then: first instruction gets cached
    _assert_instruction_contents(
        await repository.instruction(_cache_key("foo")),
        index=1,
        key=_cache_key("foo"),
    )

    # when
    clock.advance(3.0)

    # then: first instruction is cached, second gets cached
    _assert_instruction_contents(
        await repository.instruction(_cache_key("foo")),
        index=1,
        key=_cache_key("foo"),
    )

    _assert_instruction_contents(
        await repository.instruction(_cache_key("bar")),
        index=2,
        key=_cache_key("bar"),
    )

    # when
    clock.advance(7.0)

    # then: both instructions are cached
    _assert_instruction_contents(
        await repository.instruction(_cache_key("foo")),
        index=1,
        key=_cache_key("foo"),
    )

    _assert_instruction_contents(
        await repository.instruction(_cache_key("bar")),
        index=2,
        key=_cache_key("bar"),
    )

    # when
    clock.advance(0.000_001)

    # then: first instruction's cache expired, second is still cached
    _assert_instruction_contents(
        await repository.instruction(_cache_key("foo")),
        index=3,
        key=_cache_key("foo"),
    )

    _assert_instruction_contents(
        await repository.instruction(_cache_key("bar")),
        index=2,
        key=_cache_key("bar"),
    )

    # when
    clock.advance(100.0)

    # then: both instructions cache expired
    _assert_instruction_contents(
        await repository.instruction(_cache_key("foo")),
        index=4,
        key=_cache_key("foo"),
    )

    _assert_instruction_contents(
        await repository.instruction(_cache_key("bar")),
        index=5,
        key=_cache_key("bar"),
    )


def _assert_instruction_contents(
    instruction: Instruction,
    *,
    index: int,
    key: str,
) -> None:
    assert instruction.instruction == f"My instruction #{index} for {key=}"
