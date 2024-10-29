from collections.abc import Callable, Collection, Iterable
from typing import Any, Literal, Self, cast, final

from haiway import freeze

from draive.parameters.errors import ParameterValidationError
from draive.parameters.path import ParameterPath
from draive.parameters.validation import ParameterValidationContext

__all__ = [
    "ParameterRequirement",
]


@final
class ParameterRequirement[Root]:
    @classmethod
    def equal[Parameter](
        cls,
        value: Parameter,
        /,
        path: ParameterPath[Root, Parameter] | Parameter,
    ) -> Self:
        assert isinstance(  # nosec: B101
            path, ParameterPath
        ), "Prepare parameter path by using Self._.path.to.property"

        def check_equal(root: Root) -> None:
            checked: Any = cast(ParameterPath[Root, Parameter], path)(root)
            if checked != value:
                raise ParameterValidationError.invalid(
                    context=ParameterValidationContext(
                        path=cast(ParameterPath[Root, Parameter], path).components(),
                    ),
                    exception=ValueError(f"{checked} is not equal {value}"),
                )

        return cls(
            path,
            "equal",
            value,
            check=check_equal,
        )

    @classmethod
    def not_equal[Parameter](
        cls,
        value: Parameter,
        /,
        path: ParameterPath[Root, Parameter] | Parameter,
    ) -> Self:
        assert isinstance(  # nosec: B101
            path, ParameterPath
        ), "Prepare parameter path by using Self._.path.to.property"

        def check_not_equal(root: Root) -> None:
            checked: Any = cast(ParameterPath[Root, Parameter], path)(root)
            if checked == value:
                raise ParameterValidationError.invalid(
                    context=ParameterValidationContext(
                        path=cast(ParameterPath[Root, Parameter], path).components(),
                    ),
                    exception=ValueError(f"{checked} is equal {value}"),
                )

        return cls(
            path,
            "not_equal",
            value,
            check=check_not_equal,
        )

    @classmethod
    def contains[Parameter](
        cls,
        value: Parameter,
        /,
        path: ParameterPath[
            Root,
            Collection[Parameter] | tuple[Parameter, ...] | list[Parameter] | set[Parameter],
        ]
        | Collection[Parameter]
        | tuple[Parameter, ...]
        | list[Parameter]
        | set[Parameter],
    ) -> Self:
        assert isinstance(  # nosec: B101
            path, ParameterPath
        ), "Prepare parameter path by using Self._.path.to.property"

        def check_contains(root: Root) -> None:
            checked: Any = cast(ParameterPath[Root, Parameter], path)(root)
            if value not in checked:
                raise ParameterValidationError.invalid(
                    context=ParameterValidationContext(
                        path=cast(ParameterPath[Root, Parameter], path).components(),
                    ),
                    exception=ValueError(f"{checked} does not contain {value}"),
                )

        return cls(
            path,
            "contains",
            value,
            check=check_contains,
        )

    @classmethod
    def contains_any[Parameter](
        cls,
        value: Collection[Parameter],
        /,
        path: ParameterPath[
            Root,
            Collection[Parameter] | tuple[Parameter, ...] | list[Parameter] | set[Parameter],
        ]
        | Collection[Parameter]
        | tuple[Parameter, ...]
        | list[Parameter]
        | set[Parameter],
    ) -> Self:
        assert isinstance(  # nosec: B101
            path, ParameterPath
        ), "Prepare parameter path by using Self._.path.to.property"

        def check_contains_any(root: Root) -> None:
            checked: Any = cast(ParameterPath[Root, Parameter], path)(root)
            if any(element in checked for element in value):
                raise ParameterValidationError.invalid(
                    context=ParameterValidationContext(
                        path=cast(ParameterPath[Root, Parameter], path).components(),
                    ),
                    exception=ValueError(f"{checked} does not contain any of {value}"),
                )

        return cls(
            path,
            "contains_any",
            value,
            check=check_contains_any,
        )

    @classmethod
    def contained_in[Parameter](
        cls,
        value: Collection[Parameter],
        /,
        path: ParameterPath[Root, Parameter] | Parameter,
    ) -> Self:
        assert isinstance(  # nosec: B101
            path, ParameterPath
        ), "Prepare parameter path by using Self._.path.to.property"

        def check_contained_in(root: Root) -> None:
            checked: Any = cast(ParameterPath[Root, Parameter], path)(root)
            if checked not in value:
                raise ParameterValidationError.invalid(
                    context=ParameterValidationContext(
                        path=cast(ParameterPath[Root, Parameter], path).components(),
                    ),
                    exception=ValueError(f"{value} does not contain {checked}"),
                )

        return cls(
            value,
            "contained_in",
            path,
            check=check_contained_in,
        )

    def __init__(
        self,
        lhs: Any,
        operator: Literal[
            "equal",
            "not_equal",
            "contains",
            "contains_any",
            "contained_in",
            "and",
            "or",
        ],
        rhs: Any,
        check: Callable[[Root], None],
    ) -> None:
        self.lhs: Any = lhs
        self.operator: Literal[
            "equal",
            "not_equal",
            "contains",
            "contains_any",
            "contained_in",
            "and",
            "or",
        ] = operator
        self.rhs: Any = rhs
        self._check: Callable[[Root], None] = check

        freeze(self)

    def __and__(
        self,
        other: Self,
    ) -> Self:
        def check_and(root: Root) -> None:
            self.check(root)
            other.check(root)

        return self.__class__(
            self,
            "and",
            other,
            check=check_and,
        )

    def __or__(
        self,
        other: Self,
    ) -> Self:
        def check_or(root: Root) -> None:
            try:
                self.check(root)
            except ValueError:
                other.check(root)

        return self.__class__(
            self,
            "or",
            other,
            check=check_or,
        )

    def check(
        self,
        root: Root,
        /,
        *,
        raise_exception: bool = True,
    ) -> bool:
        try:
            self._check(root)
            return True

        except Exception as exc:
            if raise_exception:
                raise exc

            else:
                return False

    def filter(
        self,
        values: Iterable[Root],
    ) -> list[Root]:
        return [value for value in values if self.check(value, raise_exception=False)]
