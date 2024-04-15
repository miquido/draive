from typing import Any, Self, cast, final, overload

from draive.helpers import freeze
from draive.keypaths.component import KeyPathComponent

__all__ = [
    "KeyPath",
]


@final
class KeyPath[Root]:
    def __init__(
        self,
        root: type[Root],
        /,
        path: list[KeyPathComponent] | str | None = None,
    ) -> None:
        self._root: type[Root] = root
        self._path: list[KeyPathComponent]
        if path is None:
            self._path = []
        elif isinstance(path, str):
            self._path = [
                KeyPathComponent(
                    name=element,
                    expected=type[Any],
                )
                for element in path.split(".")
            ]
        else:
            self._path = path

        freeze(self)

    @property
    def path(self) -> str:
        return ".".join(component.name for component in self._path)

    def __str__(self) -> str:
        return f"{self._root.__qualname__}.{self.path}"

    def __getattr__(
        self,
        attribute: str,
    ) -> Self:
        return cast(
            Self,  # class is final
            KeyPath(
                self._root,
                path=[
                    *self._path,
                    KeyPathComponent(
                        name=attribute,
                        expected=type(Any),  # TODO: check type and attribute existence
                    ),
                ],
            ),
        )

    @overload
    def __call__(
        self,
        root: Root,
        /,
    ) -> Any: ...

    @overload
    def __call__[Value](
        self,
        root: Root,
        /,
        expected: type[Value],
    ) -> Value: ...

    def __call__[Value](
        self,
        root: Root,
        /,
        expected: type[Value] | None = None,
    ) -> Value:
        resolved: Any = root
        for component in self._path:
            resolved = getattr(resolved, component.name)
        if (expected := expected) and not isinstance(resolved, expected):
            raise TypeError(
                f"KeyPath '{self}' pointing to unexpected value of "
                f"'{type(resolved).__name__}' instead of '{expected.__name__}'"
            )
        return resolved
