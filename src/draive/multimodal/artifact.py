from typing import Self, final

from haiway import META_EMPTY, Meta, MetaValues

from draive.parameters import DataModel

__all__ = ("ArtifactContent",)


@final
class ArtifactContent[Artifact: DataModel = DataModel](DataModel):
    """
    Typed wrapper for a DataModel artifact with optional metadata.

    The generic parameter binds the wrapped value to a specific DataModel
    type, enabling precise typing across multimodal operations and provider
    adapters.

    Attributes
    ----------
    category : str
        Logical category label for the artifact. Used for selection/dispatch
        in multimodal pipelines and defaults to the wrapped artifact class
        name when not provided.
    artifact : Artifact
        The wrapped DataModel instance.
    hidden : bool
        Whether the artifact should be treated as non-rendering content in
        string conversions or model context. The artifact remains available
        programmatically. Defaults to False.
    meta : Meta
        Structured metadata associated with the artifact. Defaults to an
        empty metadata object.
    """

    @classmethod
    def of(
        cls,
        artifact: Artifact,
        *,
        category: str | None = None,
        hidden: bool = False,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """
        Construct an ArtifactContent instance.

        Parameters
        ----------
        artifact : Artifact
            The DataModel artifact to wrap.
        category : str | None, optional
            Logical category label for routing/selection. When None, defaults
            to ``artifact.__class__.__name__`` to provide a stable, readable
            identifier in multimodal flows.
        hidden : bool, optional
            Controls visibility in textual projections. When True, callers that
            honor this flag may omit the artifact from output conversion.
            Defaults to False.
        meta : Meta | MetaValues | None, optional
            Optional metadata to attach. Accepts an existing Meta instance,
            a mapping of metadata key-value pairs, or None to use empty meta.

        Returns
        -------
        Self
            A new ArtifactContent instance with normalized metadata.
        """

        return cls(
            category=category if category is not None else artifact.__class__.__name__,
            artifact=artifact,
            hidden=hidden,
            meta=Meta.of(meta),
        )

    category: str
    artifact: Artifact
    hidden: bool
    meta: Meta = META_EMPTY

    def to_str(self) -> str:
        """
        Produce a string representation of the artifact.

        Returns
        -------
        str
            The artifact's textual form via ``artifact.to_str()``; returns an
            empty string when ``hidden`` is True to respect non-rendering
            semantics in textual contexts.
        """
        if self.hidden:
            return ""  # empty if hidden

        return self.artifact.to_str()
