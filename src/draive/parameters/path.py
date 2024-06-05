import builtins
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, final, get_args, get_origin, overload

from draive.utils import freeze

__all__ = [
    "ParameterPath",
]


class ParameterPathComponent(ABC):
    @abstractmethod
    def path_str(
        self,
        current: str,
    ) -> str: ...

    @abstractmethod
    def resolve(
        self,
        subject: Any,
        /,
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

        self._resolve: Callable[[Root], Parameter] = resolve
        self._attribute: str = attribute

        freeze(self)

    def path_str(
        self,
        current: str,
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

        self._resolve: Callable[[Root], Parameter] = resolve
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
        *path: ParameterPathItemComponent,
    ) -> None: ...

    def __init__(
        self,
        root: type[Root],
        parameter: type[Parameter],
        *components: ParameterPathItemComponent,
    ) -> None:
        assert components or root == parameter  # nosec: B101
        self._root: type[Root] = root
        self._parameter: type[Parameter] = parameter
        self._components: tuple[ParameterPathItemComponent, ...] = components

        freeze(self)

    def components(self) -> tuple[str, ...]:
        return tuple(component.path_str() for component in self._components)

    def __str__(self) -> str:
        path: str = ""
        for component in self._components:
            path = component.path_str(path)
        return path

    def __repr__(self) -> str:
        path: str = self._root.__qualname__
        for component in self._components:
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
            annotation: Any = self._parameter.__annotations__[name]

        except (AttributeError, KeyError) as exc:
            raise AttributeError(name) from exc

        return ParameterPath[Root, Any](
            self._root,
            annotation,
            *[
                *self._components,
                ParameterPathAttributeComponent(
                    root=self._parameter,
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
        match get_origin(self._parameter) or self._parameter:
            case (
                builtins.list  # pyright: ignore[reportUnknownMemberType]
                | typing.List  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
            ):
                match get_args(self._parameter):
                    case (element_annotation,) if isinstance(key, int):
                        annotation = element_annotation

                    case other:
                        raise TypeError("Unsupported list type annotation", other)
            case (
                builtins.tuple  # pyright: ignore[reportUnknownMemberType]
                | typing.Tuple  # pyright: ignore[reportUnknownMemberType]  # noqa: UP006
            ):
                match get_args(self._parameter):
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
                match get_args(self._parameter):
                    case (builtins.str, element_annotation) if isinstance(key, str):
                        annotation = element_annotation

                    case (builtins.int, element_annotation) if isinstance(key, int):
                        annotation = element_annotation

                    case other:  # pyright: ignore[reportUnnecessaryComparison]
                        raise TypeError("Unsupported dict type annotation", other)

            case other:
                raise TypeError("Unsupported type annotation", other)

        return ParameterPath[Root, Any](
            self._root,
            annotation,
            *[
                *self._components,
                ParameterPathItemComponent(
                    root=self._parameter,
                    parameter=annotation,
                    item=key,
                ),
            ],
        )

    def __call__(
        self,
        root: Root,
    ) -> Parameter:
        assert isinstance(root, get_origin(self._root) or self._root), (  # nosec: B101
            f"ParameterPath '{self.__repr__()}' used on unexpected root of "
            f"'{type(root)}' instead of '{self._root}'"
        )

        resolved: Any = root
        for component in self._components:
            resolved = component.resolve(resolved)

        assert isinstance(resolved, get_origin(self._parameter) or self._parameter), (  # nosec: B101
            f"ParameterPath '{self.__repr__()}' pointing to unexpected value of "
            f"'{type(resolved)}' instead of '{self._parameter}'"
        )
        return resolved
