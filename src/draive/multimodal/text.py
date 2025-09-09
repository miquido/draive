from typing import Self, final

from haiway import META_EMPTY, Meta, MetaValues

from draive.parameters import DataModel

__all__ = ("TextContent",)


@final
class TextContent(DataModel):
    """
    Text block with optional metadata.

    The ``TextContent`` model represents a unit of plain text in a
    ``MultimodalContent`` sequence. The text is kept immutable and can be
    associated with structured metadata to support filtering, grouping, and
    selection during downstream processing.

    Attributes
    ----------
    text : str
        The textual content.
    meta : Meta
        Structured metadata associated with the text. Used for filtering and
        grouping operations. Defaults to an empty metadata object.
    """

    @classmethod
    def of(
        cls,
        text: str,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Construct a ``TextContent`` instance.

        Parameters
        ----------
        text : str
            The textual content value.
        meta : Meta | MetaValues | None, optional
            Optional metadata to attach. Accepts an existing ``Meta`` instance,
            a mapping of metadata key-value pairs, or ``None`` to use empty
            metadata.

        Returns
        -------
        Self
            A new ``TextContent`` instance with normalized metadata.
        """
        return cls(
            text=text,
            meta=Meta.of(meta),
        )

    text: str
    meta: Meta = META_EMPTY

    def to_str(self) -> str:
        """
        Produce the textual representation of this element.

        Returns
        -------
        str
            The raw ``text`` value without metadata.
        """
        return self.text

    def __bool__(self) -> bool:
        """
        Indicate whether the text is non-empty.

        Returns
        -------
        bool
            ``True`` if ``text`` contains at least one character, ``False``
            otherwise. Whitespace-only strings are considered truthy.
        """
        return len(self.text) > 0
