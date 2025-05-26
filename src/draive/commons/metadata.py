from collections.abc import Iterator, Mapping, Sequence
from types import EllipsisType
from typing import Any, ClassVar, Final, Self, cast, final

__all__ = (
    "META_EMPTY",
    "Meta",
    "MetaTags",
    "MetaValue",
    "MetaValues",
    "validated_meta_value",
)

type MetaValue = Mapping[str, MetaValue] | Sequence[MetaValue] | str | float | int | bool | None
type MetaValues = Mapping[str, MetaValue]
type MetaTags = Sequence[str]


@final
class Meta(Mapping[str, MetaValue]):
    _: ClassVar[Self]
    __IMMUTABLE__: ClassVar[EllipsisType] = ...

    __slots__ = ("_values",)

    def __init__(
        self,
        values: MetaValues,
        /,
    ):
        self._values: MetaValues
        if isinstance(values, Meta):  # avoid wrapping twice
            object.__setattr__(
                self,
                "_values",
                values._values,
            )

        else:
            object.__setattr__(
                self,
                "_values",
                values,
            )

    @classmethod
    def of(
        cls,
        meta: Self | MetaValues | None,
    ) -> Self:
        match meta:
            case None:
                return cast(Self, META_EMPTY)

            case Meta():
                return cast(Self, meta)

            case mapping:
                return cls({key: validated_meta_value(value) for key, value in mapping.items()})

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        /,
    ) -> Self:
        return cls({key: validated_meta_value(value) for key, value in mapping.items()})

    def to_str(self) -> str:
        return self.__str__()

    def to_mapping(
        self,
        aliased: bool = True,
    ) -> Mapping[str, Any]:
        return self._values

    @property
    def description(self) -> str | None:
        match self._values.get("description"):
            case str() as description:
                return description

            case _:
                return None

    def with_description(
        self,
        description: str,
        /,
    ) -> Self:
        return self.__class__(
            {
                **self._values,
                "description": description,
            }
        )

    @property
    def tags(self) -> MetaTags:
        match self._values.get("tags"):
            case [*tags]:
                return tuple(tag for tag in tags if isinstance(tag, str))

            case _:
                return ()

    def with_tags(
        self,
        tags: MetaTags,
        /,
    ) -> Self:
        match self._values.get("tags"):
            case [*current_tags]:
                return self.__class__(
                    {
                        **self._values,
                        "tags": (
                            *current_tags,
                            *(validated_meta_value(tag) for tag in tags if tag not in current_tags),
                        ),
                    }
                )

            case _:
                return self.__class__({**self._values, "tags": tags})

    def has_tags(
        self,
        tags: MetaTags,
        /,
    ) -> bool:
        match self._values.get("tags"):
            case [*meta_tags]:
                return all(tag in meta_tags for tag in tags)

            case _:
                return False

    def merged_with(
        self,
        values: Self | MetaValues,
        /,
    ) -> Self:
        if not values:
            return self  # do not make a copy when nothing will be updated

        return self.__class__(
            {
                **self._values,  # already validated
                **{key: validated_meta_value(value) for key, value in values.items()},
            }
        )

    def updated(
        self,
        **values: MetaValue,
    ) -> Self:
        return self.__replace__(**values)

    def __replace__(
        self,
        **values: Any,
    ) -> Self:
        return self.merged_with(values)

    def __bool__(self) -> bool:
        return bool(self._values)

    def __contains__(
        self,
        element: Any,
    ) -> bool:
        return element in self._values

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> Any:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be modified"
        )

    def __delattr__(
        self,
        name: str,
    ) -> None:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" attribute - '{name}' cannot be deleted"
        )

    def __setitem__(
        self,
        key: str,
        value: Any,
    ) -> MetaValue:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" item - '{key}' cannot be modified"
        )

    def __delitem__(
        self,
        key: str,
    ) -> MetaValue:
        raise AttributeError(
            f"Can't modify immutable {self.__class__.__qualname__},"
            f" item - '{key}' cannot be deleted"
        )

    def __getitem__(
        self,
        key: str,
    ) -> MetaValue:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __copy__(self) -> Self:
        return self  # Metadata is immutable, no need to provide an actual copy

    def __deepcopy__(
        self,
        memo: dict[int, Any] | None,
    ) -> Self:
        return self  # Metadata is immutable, no need to provide an actual copy


def validated_meta_value(value: Any) -> MetaValue:
    match value:
        case None:
            return value

        case str():
            return value

        case int():
            return value

        case float():
            return value

        case [*values]:
            return tuple(validated_meta_value(value) for value in values)

        case {**values}:
            return {key: validated_meta_value(value) for key, value in values.items()}

        case other:
            raise TypeError(f"Invalid Meta value: {type(other)}")


META_EMPTY: Final[Meta] = Meta({})
