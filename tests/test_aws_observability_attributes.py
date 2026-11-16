from typing import Any

import pytest
from haiway import Missing

from draive.aws.observability import _sanitized_attributes  # pyright: ignore[reportPrivateUsage]


class _Opaque:
    def __str__(self) -> str:
        return "opaque"


class _Secret:
    def __str__(self) -> str:
        return "secret-token"

    def __repr__(self) -> str:
        return "secret-token"


class _Broken:
    def __str__(self) -> str:
        raise RuntimeError("broken")


def test_supported_values_are_kept_and_empty_are_skipped() -> None:
    assert _sanitized_attributes(
        {
            "a": "text",
            "b": 1,
            "c": None,
            "d": Missing,
            "e": ["x", None, 2],
            "f": [],
            "g": {"k": 1.5, "n": None},
        }
    ) == {"a": "text", "b": 1, "e": ("x", 2), "g.k": 1.5}


@pytest.mark.parametrize(
    "value",
    (
        _Opaque(),
        b"bytes",
        {"nested": {"deep": 1}},
        [_Opaque(), ("a", "b")],
        {"k": _Opaque()},
        _Broken(),
    ),
)
def test_unsupported_values_never_raise(value: Any) -> None:
    result = _sanitized_attributes({"key": value})

    assert isinstance(result, dict)
    for item in result.values():
        assert isinstance(item, str | int | float | bool | tuple)


def test_unsupported_values_are_reported_by_type_only() -> None:
    assert _sanitized_attributes(
        {"a": _Opaque(), "b": [_Opaque()], "c": {"k": _Opaque()}, "d": _Broken(), "e": b"secret"}
    ) == {
        "a": "<_Opaque>",
        "b": ("<_Opaque>",),
        "c.k": "<_Opaque>",
        "d": "<_Broken>",
        "e": "<bytes>",
    }


def test_unsupported_values_do_not_leak_their_representation() -> None:
    result = _sanitized_attributes(
        {"a": _Secret(), "b": {"nested": _Secret()}, "c": [{"k": _Secret()}]}
    )

    assert "secret-token" not in repr(result)
