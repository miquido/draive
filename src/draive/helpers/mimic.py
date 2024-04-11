from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, cast

__all__ = [
    "mimic_function",
]

_Args_T = ParamSpec(
    name="_Args_T",
)
_Result_T = TypeVar(
    name="_Result_T",
)


def mimic_function(
    function: Callable[_Args_T, _Result_T],
    /,
    within: Callable[..., Any],
) -> Callable[_Args_T, _Result_T]:
    # mimic function attributes if able
    for attribute in [
        "__module__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__annotations__",
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
        Callable[_Args_T, _Result_T],
        within,
    )
