from asyncio import ALL_COMPLETED, FIRST_COMPLETED, Task, wait
from collections.abc import (
    AsyncGenerator,
    Collection,
    Generator,
    MutableSequence,
    Sequence,
)
from typing import Any, Literal, final, overload

from haiway import META_EMPTY, Meta, State, ctx, statemethod

from draive.models.tools import Toolbox
from draive.models.types import (
    ModelContext,
    ModelContextElement,
    ModelGenerating,
    ModelInput,
    ModelInstructions,
    ModelMemory,
    ModelOutput,
    ModelOutputBlock,
    ModelOutputSelection,
    ModelReasoning,
    ModelSessionOutputSelection,
    ModelSessionPreparing,
    ModelSessionScope,
    ModelStreamOutput,
    ModelToolRequest,
    ModelToolResponse,
    ModelToolsDeclaration,
)
from draive.multimodal import ArtifactContent, MultimodalContent, MultimodalContentPart, TextContent
from draive.parameters import DataModel
from draive.resources import ResourceContent, ResourceReference

__all__ = (
    "GenerativeModel",
    "RealtimeGenerativeModel",
)


@final
class GenerativeModel(State):
    """High-level generative model interface.

    This state encapsulates provider-implemented generation via ``generating`` and exposes
    convenience entrypoints for one-shot completion and tool-enabled loops. Methods are
    declared with ``@statemethod`` so they can be invoked either on an instance or on the
    class (using the active state from ``ctx``).

    Attributes
    ----------
    generating : ModelGenerating
        Provider-implemented coroutine that performs a single-turn generation. It must
        support non-streaming and streaming modes and may yield tool requests.
    """

    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instructions: ModelInstructions = "",
        tools: ModelToolsDeclaration = ModelToolsDeclaration.none,
        context: ModelContext,
        output: ModelOutputSelection = "auto",
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ModelOutput: ...

    @overload
    async def completion(
        self,
        *,
        instructions: ModelInstructions = "",
        tools: ModelToolsDeclaration = ModelToolsDeclaration.none,
        context: ModelContext,
        output: ModelOutputSelection = "auto",
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ModelOutput: ...

    @overload
    @classmethod
    async def completion(
        cls,
        *,
        instructions: ModelInstructions = "",
        tools: ModelToolsDeclaration = ModelToolsDeclaration.none,
        context: ModelContext,
        output: ModelOutputSelection = "auto",
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    @overload
    async def completion(
        self,
        *,
        instructions: ModelInstructions = "",
        tools: ModelToolsDeclaration = ModelToolsDeclaration.none,
        context: ModelContext,
        output: ModelOutputSelection = "auto",
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    @statemethod
    async def completion(
        self,
        *,
        instructions: ModelInstructions = "",
        tools: ModelToolsDeclaration = ModelToolsDeclaration.none,
        context: ModelContext,
        output: ModelOutputSelection = "auto",
        stream: bool = False,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | ModelOutput:
        """Run a single generation turn.

        Calls the provider's ``generating(...)`` once and returns either a full
        ``ModelOutput`` or a stream of ``ModelStreamOutput`` chunks when ``stream`` is
        enabled. Output is processed through ``_decoded`` or ``_decoded_stream`` to apply
        the specified output selection filtering.

        Parameters
        ----------
        instructions : ModelInstructions, optional
            System/task instructions to steer the model. Can be a string or a structured
            instructions object depending on the provider.
        tools : ModelToolsDeclaration, optional
            Tools available to the model for this turn.
        context : ModelContext
            Multimodal input context (e.g., chat history, files, images, audio).
        output : ModelOutputSelection, optional
            Desired output selection policy. ``"auto"`` preserves all content, while specific
            selections like ``"text"``, ``"image"``, ``"audio"``, ``"video"``, ``"json"``, or a
            DataModel type filter the output accordingly.
        stream : bool, optional
            When ``True``, returns an async generator yielding ``ModelStreamOutput``; when
            ``False``, returns a consolidated ``ModelOutput``.
        **extra : Any
            Provider-specific keyword arguments forwarded to ``generating(...)``.

        Returns
        -------
        ModelOutput or AsyncGenerator[ModelStreamOutput]
            Full model output with applied output selection when ``stream=False``, or a
            stream of filtered chunks when ``stream=True``.

        Raises
        ------
        ModelException
            If the underlying provider fails or the request is invalid.
        """
        if stream:
            return _decoded_stream(
                self.generating(
                    instructions=instructions,
                    tools=tools,
                    context=context,
                    output=output,
                    stream=True,
                    **extra,
                ),
                output=output,
            )

        else:
            return _decoded(
                await self.generating(
                    instructions=instructions,
                    tools=tools,
                    context=context,
                    output=output,
                    stream=False,
                    **extra,
                ),
                output=output,
            )

    @overload
    @classmethod
    async def loop(
        cls,
        *,
        instructions: ModelInstructions = "",
        toolbox: Toolbox = Toolbox.empty,
        context: MutableSequence[ModelContextElement],
        output: ModelOutputSelection = "auto",
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ModelOutput: ...

    @overload
    async def loop(
        self,
        *,
        instructions: ModelInstructions = "",
        toolbox: Toolbox = Toolbox.empty,
        context: MutableSequence[ModelContextElement],
        output: ModelOutputSelection = "auto",
        stream: Literal[False] = False,
        **extra: Any,
    ) -> ModelOutput: ...

    @overload
    @classmethod
    async def loop(
        cls,
        *,
        instructions: ModelInstructions = "",
        toolbox: Toolbox = Toolbox.empty,
        context: MutableSequence[ModelContextElement],
        output: ModelOutputSelection = "auto",
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    @overload
    async def loop(
        self,
        *,
        instructions: ModelInstructions = "",
        toolbox: Toolbox = Toolbox.empty,
        context: MutableSequence[ModelContextElement],
        output: ModelOutputSelection = "auto",
        stream: Literal[True],
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]: ...

    @statemethod
    async def loop(
        self,
        *,
        instructions: ModelInstructions = "",
        toolbox: Toolbox = Toolbox.empty,
        context: MutableSequence[ModelContextElement],
        output: ModelOutputSelection = "auto",
        stream: bool = False,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput] | ModelOutput:
        """Run a tool-use loop until completion.

        Repeatedly calls ``generating(...)`` and dispatches any ``ModelToolRequest`` blocks
        to the provided ``Toolbox``. The loop ends when a generation pass produces no tool
        requests, or when a tool responds with ``handling == "output"``. When streaming,
        yields ``ModelStreamOutput`` progressively as chunks and tool responses arrive.
        Output is processed through ``_decoded`` or ``_decoded_stream`` to apply the
        specified output selection filtering.

        Parameters
        ----------
        instructions : ModelInstructions, optional
            System/task instructions for the model across loop turns.
        toolbox : Toolbox, optional
            Collection of tools available in each turn; availability may depend on
            ``tools_turn``.
        context : MutableSequence[ModelContextElement]
            Mutable context sequence that is extended in-place with outputs and tool
            responses between turns. The final ``ModelOutput`` is appended at the end.
        output : ModelOutputSelection, optional
            Desired output selection policy. ``"auto"`` preserves all content, while specific
            selections like ``"text"``, ``"image"``, ``"audio"``, ``"video"``, ``"json"``, or a
            DataModel type filter the output accordingly.
        stream : bool, optional
            When ``True``, yields ``ModelStreamOutput`` as it is produced; when ``False``,
            returns a consolidated ``ModelOutput`` of the final result.
        **extra : Any
            Provider-specific keyword arguments forwarded to each ``generating(...)`` call.

        Returns
        -------
        ModelOutput or AsyncGenerator[ModelStreamOutput]
            Final model output with applied output selection when ``stream=False``, or a
            stream of filtered chunks when ``stream=True``.

        Raises
        ------
        ModelException
            If the underlying provider fails or the request is invalid.
        """
        if stream:
            return _decoded_stream(
                self._streaming_loop(
                    instructions=instructions,
                    toolbox=toolbox,
                    context=context,
                    output=output,
                    **extra,
                ),
                output=output,
            )

        else:
            return _decoded(
                await self._loop(
                    instructions=instructions,
                    toolbox=toolbox,
                    context=context,
                    output=output,
                    **extra,
                ),
                output=output,
            )

    async def _loop(
        self,
        *,
        instructions: ModelInstructions,
        toolbox: Toolbox,
        context: MutableSequence[ModelContextElement],
        output: ModelOutputSelection,
        **extra: Any,
    ) -> ModelOutput:
        """Run the non-streaming tool loop and return the final output.

        Executes repeated ``generating`` calls, dispatches tool requests, and merges tool
        responses back into the mutable ``context`` until a terminal output is produced.
        """
        ctx.log_debug("GenerativeModel loop started...")
        tools_turn: int = 0
        result_extension: Sequence[ModelReasoning | MultimodalContent] = ()
        while True:
            ctx.log_debug(f"...requesting generation (turn {tools_turn})...")
            generated: ModelOutput = await self.generating(
                instructions=instructions,
                tools=toolbox.available_tools_declaration(tools_turn=tools_turn),
                context=context,
                output=output,
                stream=False,
                **extra,
            )

            pending_tools: list[Task[ModelToolResponse]] = []
            for block in generated.blocks:
                if isinstance(block, ModelToolRequest):
                    pending_tools.append(ctx.spawn(toolbox.respond, block))

            if not pending_tools:
                ctx.log_debug("...loop finished...")
                result: ModelOutput = generated.updated(
                    # replace generated with forced result
                    blocks=(
                        *result_extension,
                        *(
                            block
                            for block in generated.blocks
                            if isinstance(block, MultimodalContent | ModelReasoning)
                        ),
                    )
                )
                # include final result in context
                context.append(result)
                # end the loop without tool requests - we have only the content
                return result

            ctx.log_debug("...received tool requests...")
            await wait(
                pending_tools,
                return_when=ALL_COMPLETED,
            )

            tools_result: Sequence[ModelReasoning | MultimodalContent] = ()
            tool_responses: list[ModelToolResponse] = []
            for pending in pending_tools:
                response: ModelToolResponse = pending.result()
                if response.handling == "output":
                    # skip adding to responses as we finish loop
                    # include content in tools result instead
                    tools_result = (*tools_result, response.content)

                elif response.handling == "output_extension":
                    tool_responses.append(response)
                    # include content in final result
                    result_extension = (
                        *result_extension,
                        response.content,
                    )

                else:
                    tool_responses.append(response)

            if tools_result:  # tools direct result
                ctx.log_debug("...loop finished by tools...")
                result: ModelOutput = generated.updated(
                    # replace generated with forced result
                    blocks=(
                        *result_extension,
                        *tools_result,
                    )
                )
                # include final result in context
                context.append(result)
                return result  # end the loop with tools result

            ctx.log_debug("...received tool responses...")
            # contunue to the next round with results
            context.extend(
                (
                    generated,
                    ModelInput.of(*tool_responses),
                )
            )

            tools_turn += 1  # continue with next turn

    async def _streaming_loop(  # noqa: C901, PLR0912
        self,
        *,
        instructions: ModelInstructions,
        toolbox: Toolbox,
        context: MutableSequence[ModelContextElement],
        output: ModelOutputSelection,
        **extra: Any,
    ) -> AsyncGenerator[ModelStreamOutput]:
        """Stream chunks while coordinating tool execution between turns.

        Accumulates streamed model output, forwards tool requests to the ``toolbox`` as they
        appear, yields intermediate content, and updates ``context`` until the session
        completes.
        """
        ctx.log_debug("GenerativeModel streaming loop started...")
        tools_turn: int = 0
        result_extension: Sequence[ModelReasoning | MultimodalContent] = ()
        while True:
            ctx.log_debug(f"...requesting streaming generation (turn {tools_turn})...")
            output_accumulator: list[ModelStreamOutput] = []
            pending_tools: set[Task[ModelToolResponse]] = set()
            async for chunk in self.generating(
                instructions=instructions,
                tools=toolbox.available_tools_declaration(tools_turn=tools_turn),
                context=context,
                output=output,
                stream=True,
                **extra,
            ):
                output_accumulator.append(chunk)
                if isinstance(chunk, ModelToolRequest):
                    pending_tools.add(ctx.spawn(toolbox.respond, chunk))

                else:
                    yield chunk  # stream content/reasoning parts as-is

            if not pending_tools:
                ctx.log_debug("...streaming loop finished...")
                result: Sequence[ModelReasoning | MultimodalContent] = (
                    *result_extension,
                    *(
                        block
                        for block in _merge_output(output_accumulator)
                        if isinstance(block, MultimodalContent | ModelReasoning)
                    ),
                )
                # include final result in context
                context.append(ModelOutput.of(*result))
                return  # end the loop without tool requests - we have only the content

            ctx.log_debug("...received tool requests...")
            tools_result: Sequence[ModelReasoning | MultimodalContent] = ()
            tool_responses: list[ModelToolResponse] = []
            while pending_tools:
                completed, pending_tools = await wait(
                    pending_tools,
                    return_when=FIRST_COMPLETED,
                )
                for completed_response in completed:
                    response: ModelToolResponse = completed_response.result()
                    if response.handling == "output":
                        # skip adding to responses as we finish loop
                        # include content in tools result instead
                        tools_result = (*tools_result, response.content)
                        for part in response.content.parts:
                            yield part

                    elif response.handling == "output_extension":
                        tool_responses.append(response)
                        # include content in final result
                        result_extension = (
                            *result_extension,
                            response.content,
                        )
                        for part in response.content.parts:
                            yield part

                    else:
                        tool_responses.append(response)

            if tools_result:
                ctx.log_debug("...streaming loop finished...")
                result: Sequence[ModelReasoning | MultimodalContent] = (
                    *result_extension,
                    *(
                        block
                        for block in _merge_output(output_accumulator)
                        if isinstance(block, MultimodalContent | ModelReasoning)
                    ),
                    *tools_result,
                )
                # include final result in context
                context.append(ModelOutput.of(*result))
                return  # end the loop with tools result

            ctx.log_debug("...received tool responses...")

            context.extend(
                (
                    ModelOutput.of(*_merge_output(output_accumulator)),
                    ModelInput.of(*tool_responses),
                )
            )

            tools_turn += 1  # continue with next turn

    generating: ModelGenerating


def _merge_output(  # noqa: C901, PLR0912
    output: Sequence[ModelStreamOutput],
) -> Generator[ModelOutputBlock]:
    """Merge streaming output chunks into consolidated output blocks.

    Groups consecutive multimodal content elements and preserves reasoning blocks
    as separate output blocks.

    Parameters
    ----------
    output : Sequence[ModelStreamOutput]
        Sequence of streaming output chunks to merge.

    Yields
    ------
    ModelOutputBlock
        Consolidated output blocks with merged multimodal content.
    """
    content_accumulator: list[MultimodalContentPart] = []
    reasoning_accumulator: list[MultimodalContentPart] = []
    reasoning_meta: Meta = META_EMPTY
    for element in output:
        if isinstance(element, ModelToolRequest):
            if reasoning_accumulator:
                assert not content_accumulator  # nosec: B101
                yield ModelReasoning.of(
                    MultimodalContent.of(*reasoning_accumulator),
                    meta=reasoning_meta,
                )
                reasoning_accumulator = []
                reasoning_meta = META_EMPTY

            if content_accumulator:
                assert not reasoning_accumulator  # nosec: B101
                yield MultimodalContent.of(*content_accumulator)
                content_accumulator = []

            yield element

        elif isinstance(element, ModelReasoning):
            if content_accumulator:
                yield MultimodalContent.of(*content_accumulator)
                content_accumulator = []

            if element.meta == reasoning_meta:
                reasoning_accumulator.extend(element.content.parts)

            else:
                if reasoning_accumulator:
                    yield ModelReasoning.of(
                        MultimodalContent.of(*reasoning_accumulator),
                        meta=reasoning_meta,
                    )

                reasoning_accumulator = list(element.content.parts)
                reasoning_meta = element.meta

        else:
            assert isinstance(element, MultimodalContentPart)  # nosec: B101
            if reasoning_accumulator:
                assert not content_accumulator  # nosec: B101
                yield ModelReasoning.of(
                    MultimodalContent.of(*reasoning_accumulator),
                    meta=reasoning_meta,
                )
                reasoning_accumulator = []
                reasoning_meta = META_EMPTY

            content_accumulator.append(element)

    if reasoning_accumulator:
        assert not content_accumulator  # nosec: B101
        yield ModelReasoning.of(
            MultimodalContent.of(*reasoning_accumulator),
            meta=reasoning_meta,
        )

    if content_accumulator:
        assert not reasoning_accumulator  # nosec: B101
        yield MultimodalContent.of(*content_accumulator)


@final
class RealtimeGenerativeModel(State):
    """Realtime, session-oriented generative interface.

    Wraps a provider-implemented ``session_preparing`` coroutine that creates a realtime
    session capable of bidirectional streaming and event handling. Methods are declared with
    ``@statemethod`` so they can be invoked either on an instance or on the class (using the
    active state from ``ctx``).

    Attributes
    ----------
    session_preparing : ModelSessionPreparing
        Provider-implemented coroutine that prepares and returns a ``ModelSessionScope``.
    """

    @overload
    @classmethod
    async def session(
        cls,
        *,
        instructions: ModelInstructions = "",
        memory: ModelMemory | ModelContext = (),
        tools: ModelToolsDeclaration = ModelToolsDeclaration.none,
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> ModelSessionScope: ...

    @overload
    async def session(
        self,
        *,
        instructions: ModelInstructions = "",
        memory: ModelMemory | ModelContext = (),
        tools: ModelToolsDeclaration = ModelToolsDeclaration.none,
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> ModelSessionScope: ...

    @statemethod
    async def session(
        self,
        *,
        instructions: ModelInstructions = "",
        memory: ModelMemory | ModelContext = (),
        tools: ModelToolsDeclaration = ModelToolsDeclaration.none,
        output: ModelSessionOutputSelection = "auto",
        **extra: Any,
    ) -> ModelSessionScope:
        """Prepare and open a realtime session.

        Resolves ``instructions`` (via ``InstructionsRepository``), normalizes the provided
        ``memory`` into a ``ModelMemory`` if a raw context is given, and delegates to the
        provider's ``session_preparing`` to obtain a ``ModelSessionScope``.

        Parameters
        ----------
        instructions : ModelInstructions, optional
            Instructions or an instruction reference resolvable by ``InstructionsRepository``.
        memory : ModelMemory or ModelContext, optional
            Initial memory for the session. If a plain context sequence is provided, it is
            automatically wrapped with ``ModelMemory.constant(...)`` containing the context.
        tools : ModelToolsDeclaration, optional
            Tools available in the session.
        output : ModelSessionOutputSelection, optional
            Desired output selection policy for session events (``"auto"`` chooses a sensible
            default per provider).
        **extra : Any
            Provider-specific keyword arguments forwarded to ``session_preparing(...)``.

        Returns
        -------
        ModelSessionScope
            Prepared session scope ready to be used for realtime I/O.

        Raises
        ------
        ModelException
            If the underlying provider fails or the session cannot be prepared.
        """
        session_memory: ModelMemory
        if isinstance(memory, Sequence):
            session_memory = ModelMemory.constant(*memory)

        else:
            session_memory = memory

        return await self.session_preparing(
            # TODO: FIXME: pass memory_recall.variables ??
            instructions=instructions,
            memory=session_memory,
            output=output,
            tools=tools,
            **extra,
        )

    session_preparing: ModelSessionPreparing


def _matches_modalities(
    part: MultimodalContentPart,
    *,
    allowed: Collection[Literal["text", "image", "audio", "video"]],
) -> bool:
    if not allowed:
        return False

    if "text" in allowed and isinstance(part, TextContent):
        return True

    if isinstance(part, ResourceContent | ResourceReference):
        mime_type: str = part.mime_type or ""
        if "image" in allowed and mime_type.startswith("image"):
            return True

        if "audio" in allowed and mime_type.startswith("audio"):
            return True

        if "video" in allowed and mime_type.startswith("video"):
            return True

    return False


def _decoded(  # noqa: PLR0911
    result: ModelOutput,
    /,
    *,
    output: ModelOutputSelection,
) -> ModelOutput:
    """Apply output selection filtering to a ModelOutput.

    Parameters
    ----------
    result : ModelOutput
        The raw model output to filter.
    output : ModelOutputSelection
        The output selection policy to apply.

    Returns
    -------
    ModelOutput
        Filtered model output containing only the selected content type.
    """
    match output:
        case "auto":
            return result

        case "text":
            return result.updated(blocks=(MultimodalContent.of(*result.content.texts()),))

        case "image":
            return result.updated(blocks=(MultimodalContent.of(*result.content.images()),))

        case "audio":
            return result.updated(blocks=(MultimodalContent.of(*result.content.audio()),))

        case "video":
            return result.updated(blocks=(MultimodalContent.of(*result.content.video()),))

        case selection if isinstance(selection, Collection) and not isinstance(selection, str):
            selected_parts: tuple[MultimodalContentPart, ...] = tuple(
                part
                for part in result.content.parts
                if _matches_modalities(part, allowed=set(selection))
            )
            return result.updated(blocks=(MultimodalContent.of(*selected_parts),))

        case "json":
            return result.updated(
                blocks=(
                    MultimodalContent.of(
                        ArtifactContent.of(
                            DataModel.from_json(result.content.to_str()),
                            category="json",
                        )
                    ),
                )
            )

        case model:
            assert isinstance(model, type)  # nosec: B101
            return result.updated(
                blocks=(
                    MultimodalContent.of(
                        ArtifactContent.of(
                            model.from_json(result.content.to_str()),
                            category=model.__name__,
                        )
                    ),
                )
            )


async def _decoded_stream(  # noqa: C901, PLR0912
    stream: AsyncGenerator[ModelStreamOutput],
    /,
    *,
    output: ModelOutputSelection,
) -> AsyncGenerator[ModelStreamOutput]:
    """Apply output selection filtering to a stream of ModelStreamOutput.

    Parameters
    ----------
    stream : AsyncGenerator[ModelStreamOutput]
        The raw model output stream to filter.
    output : ModelOutputSelection
        The output selection policy to apply.

    Yields
    ------
    ModelStreamOutput
        Filtered stream chunks containing only the selected content type.
    """
    match output:
        case "auto":
            async for part in stream:
                yield part

        case "text":
            async for part in stream:
                if isinstance(part, TextContent):
                    yield part

                # skip non text output

        case "image":
            async for part in stream:
                if isinstance(part, ResourceContent | ResourceReference) and (
                    (part.mime_type or "").startswith("image")
                ):
                    yield part

                # skip non image output

        case "audio":
            async for part in stream:
                if isinstance(part, ResourceContent | ResourceReference) and (
                    (part.mime_type or "").startswith("audio")
                ):
                    yield part

                # skip non audio output

        case "video":
            async for part in stream:
                if isinstance(part, ResourceContent | ResourceReference) and (
                    (part.mime_type or "").startswith("video")
                ):
                    yield part

                # skip non video output

        case selection if isinstance(selection, Collection) and not isinstance(selection, str):
            async for part in stream:
                if isinstance(part, MultimodalContentPart) and _matches_modalities(
                    part,
                    allowed=set(selection),
                ):
                    yield part

                # skip non matching output

        case "json":
            accumulator: list[str] = []
            async for part in stream:
                if isinstance(part, TextContent):
                    # we are not decoding it on the fly
                    accumulator.append(part.text)

                # skip non text output

            # provide decoded model afterwards
            yield ArtifactContent.of(
                DataModel.from_json("".join(accumulator)),
                category="json",
            )

        case model:
            assert isinstance(model, type)  # nosec: B101
            accumulator: list[str] = []
            async for part in stream:
                if isinstance(part, TextContent):
                    # we are not decoding it on the fly
                    accumulator.append(part.text)

                # skip non text output

            # provide decoded model afterwards
            yield ArtifactContent.of(
                model.from_json("".join(accumulator)),
                category=model.__name__,
            )
