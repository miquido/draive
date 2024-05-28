from os import environ, getenv
from typing import overload

__all__ = [
    "getenv_bool",
    "getenv_int",
    "getenv_float",
    "getenv_str",
    "load_env",
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


def load_env(
    path: str | None = None,
    override: bool = True,
) -> None:
    """\
    Minimalist implementation of environment variables file loader. \
    When the file is not available configuration won't be loaded.
    Allows only subset of formatting:
    - lines starting with '#' are ignored
    - other comments are not allowed
    - each element is in a new line
    - each element must be a `key=value` pair without whitespaces or additional characters
    - keys without values are ignored

    Parameters
    ----------
    path: str
        custom path to load environment variables, default is '.env'
    override: bool
        override existing variables on conflict if True, otherwise keep existing
    """

    try:
        with open(file=path or ".env") as file:
            for line in file.readlines():
                if line.startswith("#"):
                    continue  # ignore commented

                idx: int  # find where key ends
                for element in enumerate(line):
                    if element[1] == "=":
                        idx: int = element[0]
                        break
                else:  # ignore keys without assignment
                    continue

                if idx >= len(line):
                    continue  # ignore keys without values

                key: str = line[0:idx]
                value: str = line[idx + 1 :].strip()
                if value and (override or key not in environ):
                    environ[key] = value
    except FileNotFoundError:
        pass  # ignore loading if no .env available
