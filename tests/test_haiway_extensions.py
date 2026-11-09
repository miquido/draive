import asyncio
from collections.abc import Sequence

import pytest
from haiway import (
    MISSING,
    Default,
    Meta,
    Missing,
    State,
    as_dict,
    as_list,
    as_set,
    as_tuple,
    ctx,
    getenv_bool,
    getenv_float,
    getenv_int,
    getenv_str,
    is_missing,
    not_missing,
    unwrap_missing,
    without_missing,
)

import draive


@pytest.mark.parametrize(
    "name",
    (
        "ctx",
        "State",
        "Meta",
        "Default",
        "Validator",
        "Disposables",
        "MISSING",
        "as_dict",
        "as_list",
        "as_tuple",
        "as_set",
        "getenv_str",
        "getenv_int",
        "getenv_bool",
        "getenv_float",
        "cache",
        "retry",
        "timeout",
        "throttle",
        "concurrently",
        "stream_concurrently",
    ),
)
def test_draive_reexports_haiway_symbols(name: str) -> None:
    assert name in draive.__all__
    assert hasattr(draive, name)


def test_draive_exports_processing_event() -> None:
    assert "ProcessingEvent" in draive.__all__


def test_reexported_symbols_are_the_haiway_ones() -> None:
    import haiway

    assert draive.ctx is haiway.ctx
    assert draive.State is haiway.State
    assert draive.Meta is haiway.Meta


class _Config(State):
    name: str
    retries: int = Default(3)
    tags: Sequence[str] = ()


def test_state_defaults_and_updates() -> None:
    config = _Config(name="a")

    assert config.retries == 3
    assert config.tags == ()
    updated = config.updating(retries=5)
    assert updated.retries == 5
    assert config.retries == 3


def test_state_is_immutable() -> None:
    config = _Config(name="a")

    with pytest.raises((AttributeError, TypeError)):
        config.name = "b"  # pyright: ignore[reportAttributeAccessIssue]


def test_state_equality_and_mapping_round_trip() -> None:
    config = _Config(name="a", tags=("x", "y"))

    assert config == _Config(name="a", tags=("x", "y"))
    assert config != _Config(name="b", tags=("x", "y"))
    assert _Config.from_mapping(config.to_mapping()) == config


@pytest.mark.asyncio
async def test_ctx_scope_binds_state() -> None:
    async with ctx.scope("test.haiway.state", _Config(name="scoped")):
        assert ctx.state(_Config).name == "scoped"

        async with ctx.scope("test.haiway.nested", ctx.state(_Config).updating(name="nested")):
            assert ctx.state(_Config).name == "nested"

        assert ctx.state(_Config).name == "scoped"


@pytest.mark.asyncio
async def test_ctx_state_falls_back_to_default() -> None:
    class _Defaulted(State):
        value: int = 7

    async with ctx.scope("test.haiway.default"):
        assert ctx.state(_Defaulted).value == 7


@pytest.mark.asyncio
async def test_ctx_spawn_runs_within_scope() -> None:
    async def read() -> str:
        return ctx.state(_Config).name

    async with ctx.scope("test.haiway.spawn", _Config(name="spawned")):
        assert await ctx.spawn(read) == "spawned"


def test_meta_of_and_updates() -> None:
    meta = Meta.of({"a": 1})

    assert Meta.of(meta) == meta
    assert Meta.of(None) == Meta.empty
    assert meta.updating(b=2) == Meta.of({"a": 1, "b": 2})


def test_missing_helpers() -> None:
    value: str | Missing = MISSING

    assert is_missing(value)
    assert not not_missing(value)
    assert unwrap_missing(value, default="fallback") == "fallback"
    assert unwrap_missing("set", default="fallback") == "set"
    assert without_missing({"a": 1, "b": MISSING}) == {"a": 1}


def test_collection_helpers() -> None:
    assert as_list((1, 2)) == [1, 2]
    assert as_tuple([1, 2]) == (1, 2)
    assert as_set([1, 1, 2]) == {1, 2}
    assert as_dict({"a": 1}) == {"a": 1}
    assert as_list(None) is None
    assert as_tuple(None) is None


def test_getenv_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRAIVE_TEST_STR", "text")
    monkeypatch.setenv("DRAIVE_TEST_INT", "42")
    monkeypatch.setenv("DRAIVE_TEST_FLOAT", "1.5")
    monkeypatch.setenv("DRAIVE_TEST_BOOL", "true")
    monkeypatch.delenv("DRAIVE_TEST_UNSET", raising=False)

    assert getenv_str("DRAIVE_TEST_STR") == "text"
    assert getenv_int("DRAIVE_TEST_INT") == 42
    assert getenv_float("DRAIVE_TEST_FLOAT") == 1.5
    assert getenv_bool("DRAIVE_TEST_BOOL") is True
    assert getenv_str("DRAIVE_TEST_UNSET") is None
    assert getenv_str("DRAIVE_TEST_UNSET", default="fallback") == "fallback"
    assert getenv_int("DRAIVE_TEST_UNSET", default=1) == 1


@pytest.mark.asyncio
async def test_concurrently_runs_all_coroutines() -> None:
    results: list[int] = []

    async def work(value: int) -> None:
        await asyncio.sleep(0)
        results.append(value)

    await draive.concurrently((work(1), work(2), work(3)))

    assert sorted(results) == [1, 2, 3]


@pytest.mark.asyncio
async def test_timeout_and_retry_decorators() -> None:
    attempts = 0

    @draive.retry(limit=2)
    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("fail")
        return "ok"

    assert await flaky() == "ok"
    assert attempts == 2

    @draive.timeout(0.01)
    async def slow() -> None:
        await asyncio.sleep(1)

    with pytest.raises(asyncio.TimeoutError):
        await slow()


@pytest.mark.asyncio
async def test_cache_decorator_memoizes_calls() -> None:
    calls = 0

    @draive.cache(limit=2)
    async def compute(value: int) -> int:
        nonlocal calls
        calls += 1
        return value * 2

    assert await compute(2) == 4
    assert await compute(2) == 4
    assert calls == 1
    assert await compute(3) == 6
    assert calls == 2
