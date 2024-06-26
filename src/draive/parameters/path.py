import builtins
import types
import typing
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from typing import Any, final, get_args, get_origin, overload

from draive.utils import MISSING, Missing, freeze, not_missing

__all__ = [
    "ParameterPath",
]


class ParameterPathComponent(ABC):
    @abstractmethod
    def path_str(
        self,
        current: str | None = None,
    ) -> str: ...

    @abstractmethod
    def resolve(
        self,
        subject: Any,
        /,
    ) -> Any: ...

    @abstractmethod
    def assign(
        self,
        subject: Any,
        /,
        value: Any,
    ) -> Any: ...


@final
class ParameterPathAttributeComponent(ParameterPathComponent):
    def __init__[Root, Parameter](
        self,
        root: type[Root],
        parameter: type[Parameter],
        *,
        attribute: str,
    ) -> None:
        root_origin: Any = get_origin(root) or root
        parameter_origin: Any = get_origin(parameter) or parameter

        def resolve(
            subject: Root,
            /,
        ) -> Parameter:
            assert isinstance(subject, root_origin), (  # nosec: B101
                f"ParameterPathComponent used on unexpected root of "
                f"'{type(root)}' instead of '{root}' for '{attribute}'"
            )

            resolved: Any = getattr(subject, attribute)

            assert isinstance(resolved, parameter_origin), (  # nosec: B101
                f"ParameterPathComponent pointing to unexpected value of "
                f"'{type(resolved)}' instead of '{parameter}' for '{attribute}'"
            )
            return resolved

        def assign(
            subject: Root,
            /,
            value: Parameter,
        ) -> Root:
            assert isinstance(subject, root_origin), (  # nosec: B101
                f"ParameterPathComponent used on unexpected root of "
                f"'{type(root)}' instead of '{root}' for '{attribute}'"
            )
            assert isinstance(value, parameter_origin), (  # nosec: B101
                f"ParameterPathComponent assigning to unexpected value of "
                f"'{type(value)}' instead of '{parameter}' for '{attribute}'"
            )

            updated: Root
            if hasattr(subject, "updating_parameter"):  # can't check full type here
                updated = subject.updating_parameter(attribute, value=value)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]

            else:
                updated = copy(subject)
                setattr(updated, attribute, value)

            return updated  # pyright: ignore[reportUnknownVariableType]

        self._resolve: Callable[[Root], Parameter] = resolve
        self._assign: Callable[[Root, Parameter], Root] = assign
        self._attribute: str = attribute

        freeze(self)

    def path_str(
        self,
        current: str | None = None,
    ) -> str:
        if current:
            return f"{current}.{self._attribute}"
        else:
            return self._attribute

    def resolve(
        self,
        subject: Any,
        /,
    ) -> Any:
        return self._resolve(subject)

    def assign(
        self,
        subject: Any,
        /,
        value: Any,
    ) -> Any:
        return self._assign(subject, value)


@final
class ParameterPathItemComponent(ParameterPathComponent):
    def __init__[Root, Parameter](
        self,
        root: type[Root],
        parameter: type[Parameter],
        *,
        item: Any,
    ) -> None:
        root_origin: Any = get_origin(root) or root
        parameter_origin: Any = get_origin(parameter) or parameter

        def resolve(
            subject: Root,
            /,
        ) -> Parameter:
            assert isinstance(subject, root_origin), (  # nosec: B101
                f"ParameterPathComponent used on unexpected root of "
                f"'{type(root)}' instead of '{root}' for '{item}'"
            )

            resolved: Any = subject.__getitem__(item)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

            assert isinstance(resolved, parameter_origin), (  # nosec: B101
                f"ParameterPathComponent pointing to unexpected value of "
                f"'{type(resolved)}' instead of '{parameter}' for '{item}'"
            )
            return resolved

        def assign(
            subject: Root,
            /,
            value: Parameter,
        ) -> Root:
            assert isinstance(subject, root_origin), (  # nosec: B101
                f"ParameterPathComponent used on unexpected root of "
                f"'{type(root)}' instead of '{root}' for '{item}'"
            )
            assert isinstance(value, parameter_origin), (  # nosec: B101
                f"ParameterPathComponent assigning to unexpected value of "
                f"'{type(value)}' instead of '{parameter}' for '{item}'"
            )

            if hasattr(subject, "__setitem__"):  # can't check full type here
                updated: Root = copy(subject)
                updated.__setitem__(item, value)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                return updated

            elif isinstance(subject, Mapping):
                temp_dict: dict[Any, Any] = dict(subject)  # pyright: ignore[reportUnknownArgumentType]
                temp_dict[item] = value
                return subject.__class__(temp_dict)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportUnknownMemberType]

            elif isinstance(subject, Sequence):
                temp_list: list[Any] = list(subject)  # pyright: ignore[reportUnknownArgumentType]
                temp_list[item] = value
                return subject.__class__(temp_list)  # pyright: ignore[reportCallIssue, reportUnknownVariableType, reportUnknownMemberType]

            else:
                raise RuntimeError(f"Unsupported item assignment - {type(value)} in {type(root)}")

        self._resolve: Callable[[Root], Parameter] = resolve
        self._assign: Callable[[Root, Parameter], Root] = assign
        self._item: Any = item

    def path_str(
        self,
        current: str | None = None,
    ) -> str:
        if current:
            return f"{current}[{self._item}]"
        else:
            return f"[{self._item}]"

    def resolve(
        self,
        subject: Any,
        /,
    ) -> Any:
        return self._resolve(subject)

    def assign(
        self,
        subject: Any,
        /,
        value: Any,
    ) -> Any:
        return self._assign(subject, value)


@final
class ParameterPath[Root, Parameter]:
    @overload
    def __init__(
        self,
        root: type[Root],
        parameter: type[Root],
    ) -> None: ...

    @overload
    def __init__(
        self,
        root: type[Root],
        parameter: type[Parameter],
        *path: ParameterPathComponent,
    ) -> None: ...

    def __init__(
        self,
        root: type[Root],
        parameter: type[Parameter],
        *components: ParameterPathComponent,
    ) -> None:
        assert components or root == parameter  # nosec: B101
        self.__root__: type[Root] = root
        self.__parameter__: type[Parameter] = parameter
        self.__components__: tuple[ParameterPathComponent, ...] = components

        freeze(self)

    def components(self) -> tuple[str, ...]:
        return tuple(component.path_str() for component in self.__components__)

    def __str__(self) -> str:
        path: str = ""
        for component in self.__components__:
            path = component.path_str(path)
        return path

    def __repr__(self) -> str:
        path: str = self.__root__.__qualname__
        for component in self.__components__:
            path = component.path_str(path)
        return path

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        try:
            return object.__getattribute__(self, name)

        except (AttributeError, KeyError):
            pass  # continue

        assert not name.startswith(  # nosec: B101
            "_"
        ), f"Accessing private/special parameter paths ({name}) is forbidden"

        try:
            annotation: Any = self.__parameter__.__annotations__[name]

        except (AttributeError, KeyError) as exc:
            raise AttributeError(name) from exc

        return ParameterPath[Root, Any](
            self.__root__,
            annotation,
            *[
                *self.__components__,
                ParameterPathAttributeComponent(
                    root=self.__parameter__,
                    parameter=annotation,
                    attribute=name,
                ),
            ],
        )

    def __getitem__(
        self,
        key: str | int,
    ) -> Any:
        annotation: Any
        match get_origin(self.__parameter__) or self.__parameter__:
            case (
                builtins.list  # pyright: ignore[reportUnknownMemberType]
                | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
            ):
                match get_args(self.__parameter__):
                    case (element_annotation,) if isinstance(key, int):
                        annotation = element_annotation

                    case other:
                        raise TypeError("Unsupported list type annotation", other)
            case (
                builtins.tuple  # pyright: ignore[reportUnknownMemberType]
                | typing.Tuple  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
            ):
                match get_args(self.__parameter__):
                    case (element_annotation, builtins.Ellipsis | types.EllipsisType):
                        annotation = element_annotation

                    case other if isinstance(key, int):
                        annotation = other[key]

                    case other:
                        raise TypeError("Unsupported type annotation", other)
            case (
                builtins.dict  # pyright: ignore[reportUnknownMemberType]
                | typing.Dict  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
            ):
                match get_args(self.__parameter__):
                    case (builtins.str, element_annotation) if isinstance(key, str):
                        annotation = element_annotation

                    case (builtins.int, element_annotation) if isinstance(key, int):
                        annotation = element_annotation

                    case other:  # pyright: ignore[reportUnnecessaryComparison]
                        raise TypeError("Unsupported dict type annotation", other)

            case other:
                raise TypeError("Unsupported type annotation", other)

        return ParameterPath[Root, Any](
            self.__root__,
            annotation,
            *[
                *self.__components__,
                ParameterPathItemComponent(
                    root=self.__parameter__,
                    parameter=annotation,
                    item=key,
                ),
            ],
        )

    @overload
    def __call__(
        self,
        root: Root,
        /,
    ) -> Parameter: ...

    @overload
    def __call__(
        self,
        root: Root,
        /,
        updated: Parameter,
    ) -> Root: ...

    def __call__(
        self,
        root: Root,
        /,
        updated: Parameter | Missing = MISSING,
    ) -> Root | Parameter:
        assert isinstance(root, get_origin(self.__root__) or self.__root__), (  # nosec: B101
            f"ParameterPath '{self.__repr__()}' used on unexpected root of "
            f"'{type(root)}' instead of '{self.__root__}'"
        )

        if not_missing(updated):
            assert isinstance(updated, get_origin(self.__parameter__) or self.__parameter__), (  # nosec: B101
                f"ParameterPath '{self.__repr__()}' assigning to unexpected value of "
                f"'{type(updated)}' instead of '{self.__parameter__}'"
            )

            resolved: Any = root
            updates_stack: deque[tuple[Any, ParameterPathComponent]] = deque()
            for component in self.__components__:
                updates_stack.append((resolved, component))
                resolved = component.resolve(resolved)

            updated_value: Any = updated
            while updates_stack:
                subject, component = updates_stack.pop()
                updated_value = component.assign(
                    subject,
                    value=updated_value,
                )

            return updated_value

        else:
            resolved: Any = root
            for component in self.__components__:
                resolved = component.resolve(resolved)

            assert isinstance(resolved, get_origin(self.__parameter__) or self.__parameter__), (  # nosec: B101
                f"ParameterPath '{self.__repr__()}' pointing to unexpected value of "
                f"'{type(resolved)}' instead of '{self.__parameter__}'"
            )

            return resolved
