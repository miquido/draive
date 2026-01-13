from typing import Any, final, overload

from haiway import State, statemethod

from draive.models.types import (
    ModelContext,
    ModelGenerating,
    ModelInstructions,
    ModelOutputSelection,
    ModelOutputStream,
    ModelSessionOutputSelection,
    ModelSessionPreparing,
    ModelSessionScope,
    ModelTools,
)

__all__ = (
    "GenerativeModel",
    "RealtimeGenerativeModel",
)


@final
class GenerativeModel(State):
    """Typed facade for single-turn generative model completions.

    Delegates execution to a configured ``ModelGenerating`` callable while
    preserving strict input and output typing for tools, context, and output
    selection.

    Parameters
    ----------
    generating : ModelGenerating
        Async callable implementing provider-specific completion behavior.
    """

    @overload
    @classmethod
    def completion(  # pyright: ignore[reportInconsistentOverload]
        # it seems to be pyright limitation and false positive
        cls,
        *,
        instructions: ModelInstructions = "",
        tools: ModelTools = ModelTools.none,
        context: ModelContext,
        output: ModelOutputSelection = "auto",
        **extra: Any,
    ) -> ModelOutputStream: ...

    @overload
    def completion(
        self,
        *,
        instructions: ModelInstructions = "",
        tools: ModelTools = ModelTools.none,
        context: ModelContext,
        output: ModelOutputSelection = "auto",
        **extra: Any,
    ) -> ModelOutputStream: ...

    @statemethod
    def completion(
        self,
        *,
        instructions: ModelInstructions = "",
        tools: ModelTools = ModelTools.none,
        context: ModelContext,
        output: ModelOutputSelection = "auto",
        **extra: Any,
    ) -> ModelOutputStream:
        return self._generating(
            instructions=instructions,
            tools=tools,
            context=context,
            output=output,
            **extra,
        )

    _generating: ModelGenerating

    def __init__(
        self,
        generating: ModelGenerating,
    ) -> None:
        """Initialize the generative model wrapper.

        Parameters
        ----------
        generating : ModelGenerating
            Async callable used to execute completion requests.
        """
        super().__init__(_generating=generating)


@final
class RealtimeGenerativeModel(State):
    """Typed facade for realtime, session-based generative interactions.

    Delegates setup to a configured ``ModelSessionPreparing``
    callable and returns a scoped session object that encapsulates ongoing
    bidirectional model interaction.

    Parameters
    ----------
    session_preparing : ModelSessionPreparing
        Async callable implementing provider-specific realtime session setup.
    """

    @overload
    @classmethod
    async def session(
        cls,
        *,
        instructions: ModelInstructions = "",
        tools: ModelTools = ModelTools.none,
        context: ModelContext = (),
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> ModelSessionScope: ...

    @overload
    async def session(
        self,
        *,
        instructions: ModelInstructions = "",
        tools: ModelTools = ModelTools.none,
        context: ModelContext = (),
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> ModelSessionScope: ...

    @statemethod
    async def session(
        self,
        *,
        instructions: ModelInstructions = "",
        tools: ModelTools = ModelTools.none,
        context: ModelContext = (),
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> ModelSessionScope:
        """Open a realtime model session with the provided configuration.

        Parameters
        ----------
        instructions : ModelInstructions, default=""
            System-level instructions applied to the session.
        tools : ModelTools, default=ModelTools.none
            Tooling configuration available during session interaction.
        context : ModelContext, default=()
            Initial context seeded into the session.
        output : ModelSessionOutputSelection, default="auto"
            Output selection strategy for realtime session responses.
        **extra : Any
            Provider-specific keyword arguments forwarded unchanged.

        Returns
        -------
        ModelSessionScope
            Scoped realtime session handle.

        Raises
        ------
        Exception
            Propagates exceptions raised by the configured session preparer.
        """
        return await self._session_preparing(
            instructions=instructions,
            tools=tools,
            context=context,
            output=output,
            **extra,
        )

    _session_preparing: ModelSessionPreparing

    def __init__(
        self,
        session_preparing: ModelSessionPreparing,
    ) -> None:
        """Initialize the realtime generative model wrapper.

        Parameters
        ----------
        session_preparing : ModelSessionPreparing
            Async callable used to initialize realtime model sessions.
        """
        super().__init__(_session_preparing=session_preparing)
