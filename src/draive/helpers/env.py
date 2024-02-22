from os import getenv
from typing import overload

__all__ = [
    "getenv_bool",
    "getenv_int",
    "getenv_float",
    "getenv_str",
]


@overload
def getenv_bool(key: str) -> bool | None:
    ...


@overload
def getenv_bool(key: str, default: bool) -> bool:
    ...


def getenv_bool(
    key: str,
    default: bool | None = None,
) -> bool | None:
    if value := getenv(key=key):
        return value.lower() in ("true", "1", "t")
    else:
        return default


@overload
def getenv_int(key: str) -> int | None:
    ...


@overload
def getenv_int(key: str, default: int) -> int:
    ...


def getenv_int(
    key: str,
    default: int | None = None,
) -> int | None:
    if value := getenv(key=key):
        return int(value)

    else:
        return default


@overload
def getenv_float(key: str) -> float | None:
    ...


@overload
def getenv_float(key: str, default: float) -> float:
    ...


def getenv_float(
    key: str,
    default: float | None = None,
) -> float | None:
    if value := getenv(key=key):
        return float(value)

    else:
        return default


@overload
def getenv_str(key: str) -> str | None:
    ...


@overload
def getenv_str(key: str, default: str) -> str:
    ...


def getenv_str(
    key: str,
    default: str | None = None,
) -> str | None:
    if value := getenv(key=key):
        return value
    else:
        return default
