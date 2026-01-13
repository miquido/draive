import json
from datetime import datetime
from typing import Self, final, overload
from uuid import UUID

from haiway import BasicObject, BasicValue, Meta, MetaValues, State

__all__ = ("ArtifactContent",)


@final
class ArtifactContent(State, serializable=True):
    """Typed wrapper for structured artifact payloads used in multimodal content.

    This state keeps arbitrary JSON-like data (`artifact`) together with classification
    metadata (`category`, `meta`) and visibility control (`hidden`). It also provides
    typed accessors for common scalar value conversions.

    Parameters
    ----------
    category : str
        Logical artifact kind used for routing, presentation, or downstream decoding.
    artifact : BasicObject
        Serialized artifact payload represented as a JSON-like mapping.
    hidden : bool, default=False
        Whether the artifact should be suppressed in string rendering.
    meta : Meta, default=Meta.empty
        Additional metadata associated with the artifact content.
    """

    @overload
    @classmethod
    def of(
        cls,
        artifact: State,
        *,
        category: str | None = None,
        hidden: bool = False,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @overload
    @classmethod
    def of(
        cls,
        artifact: BasicObject,
        *,
        category: str,
        hidden: bool = False,
        meta: Meta | MetaValues | None = None,
    ) -> Self: ...

    @classmethod
    def of(
        cls,
        artifact: State | BasicObject,
        *,
        category: str | None = None,
        hidden: bool = False,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create artifact content from either a state object or a raw mapping.

        Parameters
        ----------
        artifact : State | BasicObject
            Source artifact. `State` inputs are serialized via `to_mapping`.
        category : str | None, optional
            Artifact category. Required for raw mappings and optional for states.
            When omitted for states, the state class name is used.
        hidden : bool, default=False
            Whether the content should be hidden in string representation.
        meta : Meta | MetaValues | None, optional
            Metadata attached to the resulting artifact content.

        Returns
        -------
        Self
            New `ArtifactContent` instance.
        """
        if isinstance(artifact, ArtifactContent):
            return cls(
                category=category if category is not None else artifact.category,
                artifact=artifact.artifact,
                hidden=hidden,
                meta=Meta.of(meta),
            )

        elif isinstance(artifact, State):
            return cls(
                category=category if category is not None else artifact.__class__.__name__,
                artifact=artifact.to_mapping(recursive=True),
                hidden=hidden,
                meta=Meta.of(meta),
            )

        else:
            assert category is not None  # nosec: B101
            return cls(
                category=category,
                artifact=artifact,
                hidden=hidden,
                meta=Meta.of(meta),
            )

    category: str
    artifact: BasicObject
    hidden: bool = False
    meta: Meta = Meta.empty

    def to_str(self) -> str:
        """Render artifact payload as formatted JSON text.

        Returns
        -------
        str
            Empty string when `hidden` is `True`; otherwise an indented JSON string.
        """
        if self.hidden:
            return ""  # empty if hidden

        # defaults to json format
        return json.dumps(
            self.artifact,
            indent=2,
        )

    def to_state[Content: State](
        self,
        content: type[Content],
    ) -> Content:
        """Deserialize the stored artifact payload into a typed state object.

        Parameters
        ----------
        content : type[Content]
            Target state type used to decode the payload.

        Returns
        -------
        Content
            Decoded state instance.
        """
        return content.from_mapping(self.artifact)

    @overload
    def get_uuid(
        self,
        key: str,
    ) -> UUID | None: ...

    @overload
    def get_uuid(
        self,
        key: str,
        *,
        default: UUID,
    ) -> UUID: ...

    def get_uuid(
        self,
        key: str,
        *,
        default: UUID | None = None,
    ) -> UUID | None:
        """Read a UUID value from the artifact mapping.

        Parameters
        ----------
        key : str
            Mapping key to resolve.
        default : UUID | None, optional
            Value returned when the key is absent.

        Returns
        -------
        UUID | None
            Parsed UUID for present keys, otherwise `default`.

        Raises
        ------
        TypeError
            If the stored value is not a string.
        ValueError
            If the stored string is not a valid UUID representation.
        """
        value: BasicValue | None = self.artifact.get(key)
        if value is None:
            return default

        if not isinstance(value, str):
            raise TypeError(f"Unexpected value '{type(value).__name__}' for {key}, expected 'str'")

        return UUID(value)

    def with_uuid(
        self,
        key: str,
        *,
        value: UUID,
    ) -> Self:
        """Return a copy with a UUID stored under the selected key.

        Parameters
        ----------
        key : str
            Mapping key to update.
        value : UUID
            UUID value to serialize.

        Returns
        -------
        Self
            New artifact content instance with the updated payload.
        """
        return self.__class__(
            category=self.category,
            artifact={
                **self.artifact,
                key: str(value),
            },
            hidden=self.hidden,
            meta=self.meta,
        )

    @overload
    def get_datetime(
        self,
        key: str,
    ) -> datetime | None: ...

    @overload
    def get_datetime(
        self,
        key: str,
        *,
        default: datetime,
    ) -> datetime: ...

    def get_datetime(
        self,
        key: str,
        *,
        default: datetime | None = None,
    ) -> datetime | None:
        """Read an ISO-8601 datetime value from the artifact mapping.

        Parameters
        ----------
        key : str
            Mapping key to resolve.
        default : datetime | None, optional
            Value returned when the key is absent.

        Returns
        -------
        datetime | None
            Parsed datetime for present keys, otherwise `default`.

        Raises
        ------
        TypeError
            If the stored value is not a string.
        ValueError
            If the stored string is not a valid ISO-8601 datetime.
        """
        value: BasicValue | None = self.artifact.get(key)
        if value is None:
            return default

        if not isinstance(value, str):
            raise TypeError(f"Unexpected value '{type(value).__name__}' for {key}, expected 'str'")

        return datetime.fromisoformat(value)

    def with_datetime(
        self,
        key: str,
        *,
        value: datetime,
    ) -> Self:
        """Return a copy with a datetime stored under the selected key.

        Parameters
        ----------
        key : str
            Mapping key to update.
        value : datetime
            Datetime value to serialize with `isoformat()`.

        Returns
        -------
        Self
            New artifact content instance with the updated payload.
        """
        return self.__class__(
            category=self.category,
            artifact={
                **self.artifact,
                key: value.isoformat(),
            },
            hidden=self.hidden,
            meta=self.meta,
        )

    @overload
    def get_str(
        self,
        key: str,
    ) -> str | None: ...

    @overload
    def get_str(
        self,
        key: str,
        *,
        default: str,
    ) -> str: ...

    def get_str(
        self,
        key: str,
        *,
        default: str | None = None,
    ) -> str | None:
        """Read a string value from the artifact mapping.

        Parameters
        ----------
        key : str
            Mapping key to resolve.
        default : str | None, optional
            Value returned when the key is absent.

        Returns
        -------
        str | None
            String value for present keys, otherwise `default`.

        Raises
        ------
        TypeError
            If the stored value is not a string.
        """
        value: BasicValue | None = self.artifact.get(key)
        if value is None:
            return default

        if not isinstance(value, str):
            raise TypeError(f"Unexpected value '{type(value).__name__}' for {key}, expected 'str'")

        return value

    @overload
    def get_int(
        self,
        key: str,
    ) -> int | None: ...

    @overload
    def get_int(
        self,
        key: str,
        *,
        default: int,
    ) -> int: ...

    def get_int(
        self,
        key: str,
        *,
        default: int | None = None,
    ) -> int | None:
        """Read an integer value from the artifact mapping.

        Parameters
        ----------
        key : str
            Mapping key to resolve.
        default : int | None, optional
            Value returned when the key is absent.

        Returns
        -------
        int | None
            Integer value for present keys, otherwise `default`.

        Raises
        ------
        TypeError
            If the stored value is not an integer.
        """
        value: BasicValue | None = self.artifact.get(key)
        if value is None:
            return default

        if not isinstance(value, int):
            raise TypeError(f"Unexpected value '{type(value).__name__}' for {key}, expected 'int'")

        return value

    @overload
    def get_float(
        self,
        key: str,
    ) -> float | None: ...

    @overload
    def get_float(
        self,
        key: str,
        *,
        default: float,
    ) -> float: ...

    def get_float(
        self,
        key: str,
        *,
        default: float | None = None,
    ) -> float | None:
        """Read a float value from the artifact mapping.

        Parameters
        ----------
        key : str
            Mapping key to resolve.
        default : float | None, optional
            Value returned when the key is absent.

        Returns
        -------
        float | None
            Float value for present keys, otherwise `default`.

        Raises
        ------
        TypeError
            If the stored value is not a float.
        """
        value: BasicValue | None = self.artifact.get(key)
        if value is None:
            return default

        if not isinstance(value, float):
            raise TypeError(
                f"Unexpected value '{type(value).__name__}' for {key}, expected 'float'"
            )

        return value

    @overload
    def get_bool(
        self,
        key: str,
    ) -> bool | None: ...

    @overload
    def get_bool(
        self,
        key: str,
        *,
        default: bool,
    ) -> bool: ...

    def get_bool(
        self,
        key: str,
        *,
        default: bool | None = None,
    ) -> bool | None:
        """Read a boolean value from the artifact mapping.

        Parameters
        ----------
        key : str
            Mapping key to resolve.
        default : bool | None, optional
            Value returned when the key is absent.

        Returns
        -------
        bool | None
            Boolean value for present keys, otherwise `default`.

        Raises
        ------
        TypeError
            If the stored value is not a boolean.
        """
        value: BasicValue | None = self.artifact.get(key)
        if value is None:
            return default

        if not isinstance(value, bool):
            raise TypeError(f"Unexpected value '{type(value).__name__}' for {key}, expected 'bool'")

        return value

    def __bool__(self) -> bool:
        """Return whether the artifact payload contains any data.

        Returns
        -------
        bool
            `True` when `artifact` is non-empty, otherwise `False`.
        """
        return bool(self.artifact)
