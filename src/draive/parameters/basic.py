__all__ = [
    "BasicValue",
]

type BasicValue = dict[str, "BasicValue"] | list["BasicValue"] | str | float | int | bool | None
