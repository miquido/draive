from collections.abc import Iterable, MutableSequence, Sequence
from itertools import chain
from typing import Any, ClassVar, Self

from haiway import Meta, MetaValues, Paginated, Pagination, State

from draive.conversation.types import (
    ConversationMemoryFetching,
    ConversationMemoryRecalling,
    ConversationMemoryRemembering,
    ConversationTurn,
)
from draive.models import ModelContext

__all__ = ("ConversationMemory",)


class ConversationMemory(State):
    disabled: ClassVar[Self]  # defined after the class

    @classmethod
    def constant(
        cls,
        turns: Iterable[ConversationTurn],
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        memory: Sequence[ConversationTurn] = tuple(turns)

        async def fetch(
            pagination: Pagination,
            **extra: Any,
        ) -> Paginated[ConversationTurn]:
            _ = extra
            page_items: Sequence[ConversationTurn]
            if pagination.limit <= 0:
                page_items = ()

            elif pagination.token is None:
                page_items = tuple(memory[-pagination.limit :])

            else:
                raise ValueError("Volatile conversation memory does not support pagination tokens")

            return Paginated[ConversationTurn].of(
                page_items,
                pagination=pagination.with_token(None),
            )

        async def recall(
            pagination: Pagination | None = None,
            **extra: Any,
        ) -> ModelContext:
            _ = extra
            turns: Sequence[ConversationTurn]
            if pagination is None:
                turns = tuple(memory)

            elif pagination.limit <= 0:
                turns = ()

            else:
                turns = tuple(memory[-pagination.limit :])

            return tuple(chain.from_iterable(turn.to_model_context() for turn in turns))

        async def remember(
            turns: Iterable[ConversationTurn],
            **extra: Any,
        ) -> None:
            pass  # ignoring

        return cls(
            fetching=fetch,
            recalling=recall,
            remembering=remember,
            meta=Meta.of(meta),
        )

    @classmethod
    def volatile(
        cls,
        *,
        meta: Meta | MetaValues | None = None,
    ) -> Self:
        memory: MutableSequence[ConversationTurn] = []

        async def fetch(
            pagination: Pagination,
            **extra: Any,
        ) -> Paginated[ConversationTurn]:
            _ = extra
            page_items: Sequence[ConversationTurn]
            if pagination.limit <= 0:
                page_items = ()

            elif pagination.token is None:
                page_items = tuple(memory[-pagination.limit :])

            else:
                raise ValueError("Volatile conversation memory does not support pagination tokens")

            return Paginated[ConversationTurn].of(
                page_items,
                pagination=pagination.with_token(None),
            )

        async def recall(
            pagination: Pagination | None = None,
            **extra: Any,
        ) -> ModelContext:
            _ = extra
            turns: Sequence[ConversationTurn]
            if pagination is None:
                turns = tuple(memory)

            elif pagination.limit <= 0:
                turns = ()

            else:
                turns = tuple(memory[-pagination.limit :])

            return tuple(chain.from_iterable(turn.to_model_context() for turn in turns))

        async def remember(
            turns: Iterable[ConversationTurn],
            **extra: Any,
        ) -> None:
            memory.extend(turns)

        return cls(
            fetching=fetch,
            recalling=recall,
            remembering=remember,
            meta=Meta.of(meta),
        )

    async def fetch(
        self,
        pagination: Pagination = Pagination.of(limit=32),  # noqa: B008 - immutable
        **extra: Any,
    ) -> Paginated[ConversationTurn]:
        return await self._fetching(pagination, **extra)

    async def recall(
        self,
        pagination: Pagination | None = None,
        **extra: Any,
    ) -> ModelContext:
        return await self._recalling(pagination=pagination, **extra)

    async def remember(
        self,
        *turns: ConversationTurn,
        **extra: Any,
    ) -> None:
        if not turns:
            return

        await self._remembering(turns, **extra)

    _fetching: ConversationMemoryFetching
    _recalling: ConversationMemoryRecalling
    _remembering: ConversationMemoryRemembering
    meta: Meta

    def __init__(
        self,
        fetching: ConversationMemoryFetching,
        recalling: ConversationMemoryRecalling,
        remembering: ConversationMemoryRemembering,
        meta: Meta = Meta.empty,
    ) -> None:
        super().__init__(
            _fetching=fetching,
            _recalling=recalling,
            _remembering=remembering,
            meta=meta,
        )


async def _fetch(
    pagination: Pagination,
    **extra: Any,
) -> Paginated[ConversationTurn]:
    return Paginated[ConversationTurn].of(
        (),
        pagination=pagination.with_token(None),
    )


async def _recall(
    pagination: Pagination | None = None,
    **extra: Any,
) -> ModelContext:
    _ = pagination, extra
    return ()


async def _remember(
    turns: Sequence[ConversationTurn],
    **extra: Any,
) -> None:
    pass


ConversationMemory.disabled = ConversationMemory(
    fetching=_fetch,
    recalling=_recall,
    remembering=_remember,
)
