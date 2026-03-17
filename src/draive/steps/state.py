from collections.abc import Mapping
from typing import Literal, Self, overload

from haiway import State

from draive.models import ModelContext
from draive.models.types import ModelContextElement
from draive.multimodal import ArtifactContent

__all__ = ("StepState",)


class StepState(State, serializable=True):
    """Immutable state container used by step pipelines.

    The state combines conversational model context with typed artifacts
    produced by previously executed steps. Artifacts are stored as
    :class:`draive.multimodal.ArtifactContent` instances keyed by logical names.

    Attributes
    ----------
    context : ModelContext
        Ordered model context elements accumulated for downstream processing.
    artifacts : Mapping[str, ArtifactContent]
        Named artifact payloads available to subsequent steps.
    """

    @classmethod
    def of(
        cls,
        context: ModelContext = (),
        /,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> Self:
        """Create step state from context and typed artifacts.

        Positional artifacts are stored under their class names. Keyword
        artifacts are stored under explicit keys.

        Parameters
        ----------
        context : ModelContext, optional
            Initial model context for the step state.
        *artifacts : State
            Positional artifacts keyed by each artifact class name.
        **keyed_artifacts : State
            Artifacts stored under explicit user-provided keys.

        Returns
        -------
        Self
            Newly constructed step state instance.
        """
        return cls(
            context=context,
            artifacts={
                **{item.__class__.__name__: ArtifactContent.of(item) for item in artifacts},
                **{key: ArtifactContent.of(value) for key, value in keyed_artifacts.items()},
            },
        )

    context: ModelContext = ()
    artifacts: Mapping[str, ArtifactContent]

    @overload
    def get[Content: State](
        self,
        content: type[Content],
        /,
    ) -> Content | None: ...

    @overload
    def get[Content: State](
        self,
        content: type[Content],
        /,
        *,
        key: str,
    ) -> Content | None: ...

    @overload
    def get[Content: State](
        self,
        content: type[Content],
        /,
        *,
        key: str | None = None,
        required: Literal[True],
    ) -> Content: ...

    @overload
    def get[Content: State](
        self,
        content: type[Content],
        /,
        *,
        key: str | None = None,
        default: Content,
    ) -> Content: ...

    def get[Content: State](
        self,
        content: type[Content],
        /,
        *,
        key: str | None = None,
        default: Content | None = None,
        required: bool = False,
    ) -> Content | None:
        """Return an artifact converted to the requested state type.

        Parameters
        ----------
        content : type[Content]
            Expected state type to deserialize from artifact storage.
        key : str | None, optional
            Custom artifact key. Defaults to ``content.__name__``.
        default : Content | None, optional
            Value returned when the artifact is missing and not required.
        required : bool, optional
            Whether missing artifacts should raise ``KeyError``.

        Returns
        -------
        Content | None
            Resolved artifact converted to ``content`` or ``default``/``None``
            when unavailable.

        Raises
        ------
        KeyError
            If the artifact is missing and ``required`` is ``True``.
        """
        lookup_key: str = key or content.__name__
        current: ArtifactContent | None = self.artifacts.get(lookup_key)
        if current is not None:
            return current.to_state(content)

        elif default is not None:
            return default

        elif required:
            raise KeyError(f"Artifact {lookup_key} is not available")

        else:
            return None

    def updating_artifacts(
        self,
        *artifacts: State,
        **keyed_artifacts: State,
    ) -> Self:
        """Return a copy of state with additional or replaced artifacts.

        Positional artifacts are keyed by class name. Keyword artifacts are
        merged under explicit names.

        Parameters
        ----------
        *artifacts : State
            Artifacts stored under their class names.
        **keyed_artifacts : State
            Artifacts stored under explicit keys.

        Returns
        -------
        Self
            Updated state instance. Returns ``self`` when no updates are given.
        """
        if not artifacts and not keyed_artifacts:
            return self

        return self.updating(
            artifacts={
                **self.artifacts,
                **{
                    artifact.__class__.__name__: ArtifactContent.of(artifact)
                    for artifact in artifacts
                },
                **{key: ArtifactContent.of(value) for key, value in keyed_artifacts.items()},
            }
        )

    def appending_context(
        self,
        *elements: ModelContextElement,
    ) -> Self:
        """Return a copy of state with additional context elements appended.

        Parameters
        ----------
        *elements : ModelContextElement
            Context elements appended in order to existing context.

        Returns
        -------
        Self
            Updated state instance. Returns ``self`` when no elements are given.
        """
        if not elements:
            return self

        return self.updating(
            context=(*self.context, *elements),
        )

    def replacing_context(
        self,
        context: ModelContext,
    ) -> Self:
        """Return a copy of state with model context replaced.

        Parameters
        ----------
        context : ModelContext
            New context sequence replacing current context.

        Returns
        -------
        Self
            Updated state instance. Returns ``self`` when ``context`` is empty.
        """

        return self.updating(context=context)
