from collections.abc import AsyncIterable, Generator, MutableSequence, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, Self, final, runtime_checkable
from uuid import UUID, uuid4

from haiway import Meta, MetaValues, Paginated, Pagination, State

from draive.models import (
    ModelContext,
    ModelContextElement,
    ModelInput,
    ModelInputBlock,
    ModelOutput,
    ModelOutputBlock,
    ModelReasoning,
    ModelReasoningChunk,
    ModelToolRequest,
    ModelToolResponse,
)
from draive.multimodal import ArtifactContent, MultimodalContent, MultimodalContentPart
from draive.tools.types import ToolEvent

__all__ = (
    "ConversationAssistantTurn",
    "ConversationEvent",
    "ConversationInputChunk",
    "ConversationInputStream",
    "ConversationMemoryFetching",
    "ConversationMemoryRecalling",
    "ConversationMemoryRemembering",
    "ConversationOutputChunk",
    "ConversationOutputStream",
    "ConversationTurn",
    "ConversationUserTurn",
)


@final
class ConversationEvent(State, serializable=True):
    @classmethod
    def of(
        cls,
        event: str,
        *,
        content: State | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            event=event,
            created=datetime.now(UTC),
            content=ArtifactContent.of(content) if content is not None else None,
            meta=Meta.of(meta),
        )

    @classmethod
    def tool_request(
        cls,
        request: ModelToolRequest,
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a tool request event."""
        return cls(
            event="tool_request",
            created=datetime.now(UTC),
            content=ArtifactContent.of(request),
            meta=Meta.of(meta),
        )

    @classmethod
    def tool_response(
        cls,
        response: ModelToolResponse,
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a tool response event."""
        return cls(
            event="tool_response",
            created=datetime.now(UTC),
            content=ArtifactContent.of(response),
            meta=Meta.of(meta),
        )

    @classmethod
    def tool_event(
        cls,
        event: ToolEvent,
        /,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        """Create a tool tool event."""
        return cls(
            event="tool_event",
            created=datetime.now(UTC),
            content=ArtifactContent.of(event),
            meta=Meta.of(meta),
        )

    event: str
    created: datetime
    content: ArtifactContent | None = None
    meta: Meta = Meta.empty


@final
class ConversationUserTurn(State, serializable=True):
    @classmethod
    def of(
        cls,
        *content: MultimodalContent,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            turn="user",
            created=created if created is not None else datetime.now(UTC),
            content=content,
            meta=Meta.of(meta),
        )

    identifier: UUID
    created: datetime
    turn: Literal["user"] = "user"
    content: Sequence[MultimodalContent]
    meta: Meta = Meta.empty

    def to_model_context(self) -> Generator[ModelContextElement]:
        yield ModelInput.of(
            *self.content,
            meta=self.meta,
        )

    def __bool__(self) -> bool:
        return bool(self.content)


@final
class ConversationAssistantTurn(State, serializable=True):
    @classmethod
    def of(
        cls,
        *content: MultimodalContent | ModelReasoning | ConversationEvent,
        identifier: UUID | None = None,
        created: datetime | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        return cls(
            identifier=identifier if identifier is not None else uuid4(),
            turn="assistant",
            created=created if created is not None else datetime.now(UTC),
            content=content,
            meta=Meta.of(meta),
        )

    identifier: UUID
    created: datetime
    turn: Literal["assistant"] = "assistant"
    content: Sequence[MultimodalContent | ModelReasoning | ConversationEvent]
    meta: Meta = Meta.empty

    def to_model_context(self) -> Generator[ModelContextElement]:
        input_accumulator: MutableSequence[ModelInputBlock] = []
        output_accumulator: MutableSequence[ModelOutputBlock] = []
        for element in self.content:
            if isinstance(element, ConversationEvent):
                if tool_request := _tool_request_from_event(element):
                    if input_accumulator:
                        yield ModelInput.of(
                            *input_accumulator,
                            meta=self.meta,
                        )
                        input_accumulator.clear()

                    output_accumulator.append(tool_request)

                elif tool_response := _tool_response_from_event(element):
                    if output_accumulator:
                        yield ModelOutput.of(
                            *output_accumulator,
                            meta=self.meta,
                        )
                        output_accumulator.clear()

                    input_accumulator.append(tool_response)

            else:
                if input_accumulator:
                    yield ModelInput.of(
                        *input_accumulator,
                        meta=self.meta,
                    )
                    input_accumulator.clear()

                output_accumulator.append(element)

        if input_accumulator:
            yield ModelInput.of(
                *input_accumulator,
                meta=self.meta,
            )

        if output_accumulator:
            yield ModelOutput.of(
                *output_accumulator,
                meta=self.meta,
            )

    def __bool__(self) -> bool:
        return bool(self.content)


ConversationTurn = ConversationUserTurn | ConversationAssistantTurn
ConversationInputChunk = MultimodalContentPart | ConversationEvent
ConversationInputStream = AsyncIterable[ConversationInputChunk]
ConversationOutputChunk = MultimodalContentPart | ModelReasoningChunk | ConversationEvent
ConversationOutputStream = AsyncIterable[ConversationOutputChunk]


@runtime_checkable
class ConversationMemoryRemembering(Protocol):
    async def __call__(
        self,
        turns: Sequence[ConversationTurn],
        **extra: Any,
    ) -> None: ...


@runtime_checkable
class ConversationMemoryRecalling(Protocol):
    async def __call__(
        self,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> ModelContext: ...


@runtime_checkable
class ConversationMemoryFetching(Protocol):
    async def __call__(
        self,
        pagination: Pagination,
        **extra: Any,
    ) -> Paginated[ConversationTurn]: ...


def _tool_response_from_event(
    event: ConversationEvent,
    /,
) -> ModelToolResponse | None:
    if event.content is None:
        return None

    if event.event != "tool_response":
        return None

    return event.content.to_state(ModelToolResponse)


def _tool_request_from_event(
    event: ConversationEvent,
    /,
) -> ModelToolRequest | None:
    if event.content is None:
        return None

    if event.event != "tool_request":
        return None

    return event.content.to_state(ModelToolRequest)
