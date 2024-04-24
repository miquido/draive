from collections.abc import Callable
from typing import Any, cast

__all__ = [
    "mimic_function",
]


def mimic_function[**Args, Result](
    function: Callable[Args, Result],
    /,
    within: Callable[..., Any],
) -> Callable[Args, Result]:
    # mimic function attributes if able
    for attribute in [
        "__module__",
        "__name__",
        "__qualname__",
        "__annotations__",
        "__defaults__",
        "__kwdefaults__",
        "__doc__",
    ]:
        try:
            setattr(
                within,
                attribute,
                getattr(
                    function,
                    attribute,
                ),
            )

        except AttributeError:
            pass
    try:
        within.__dict__.update(function.__dict__)
    except AttributeError:
        pass

    return cast(
        Callable[Args, Result],
        within,
    )
