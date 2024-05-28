from collections.abc import Callable
from typing import Any, cast, overload

__all__ = [
    "mimic_function",
]


@overload
def mimic_function[**Args, Result](
    function: Callable[Args, Result],
    /,
    within: Callable[..., Any],
) -> Callable[Args, Result]: ...


@overload
def mimic_function[**Args, Result](
    function: Callable[Args, Result],
    /,
) -> Callable[[Callable[..., Any]], Callable[Args, Result]]: ...


def mimic_function[**Args, Result](
    function: Callable[Args, Result],
    /,
    within: Callable[..., Result] | None = None,
) -> Callable[[Callable[..., Result]], Callable[Args, Result]] | Callable[Args, Result]:
    def mimic(
        target: Callable[..., Result],
    ) -> Callable[Args, Result]:
        # mimic function attributes if able
        for attribute in (
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
            "__type_params__",
            "__defaults__",
            "__kwdefaults__",
            "__globals__",
        ):
            try:
                setattr(
                    target,
                    attribute,
                    getattr(
                        function,
                        attribute,
                    ),
                )

            except AttributeError:
                pass
        try:
            target.__dict__.update(function.__dict__)
        except AttributeError:
            pass

        setattr(  # noqa: B010 - mimic functools.wraps behavior for correct signature checks
            target,
            "__wrapped__",
            function,
        )

        return cast(
            Callable[Args, Result],
            target,
        )

    if target := within:
        return mimic(target)
    else:
        return mimic
