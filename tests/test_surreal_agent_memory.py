import asyncio
from collections.abc import AsyncIterable, Mapping, Sequence
from typing import Any
from uuid import uuid4

import pytest
from haiway import ctx
from surrealdb import RecordID

import draive.surreal.agent_memory as surreal_agent_memory
from draive.agents import AgentThread
from draive.embedding import Embedded, TextEmbedding
from draive.generation import ModelGeneration
from draive.models import (
    GenerativeModel,
    ModelInput,
    ModelOutput,
    ModelOutputChunk,
    ModelToolRequest,
)
from draive.multimodal import (
    ArtifactContent,
    MultimodalContent,
    Template,
    TemplatesRepository,
    TextContent,
)
from draive.surreal.agent_memory import (
    EntityExtraction,
    EntityOperation,
    MemoryLayer,
    RelationExtraction,
    RelationOperation,
    _apply_relation_operations,  # pyright: ignore[reportPrivateUsage]
    _classify_entity_operations,  # pyright: ignore[reportPrivateUsage]
    _entity_record_id,  # pyright: ignore[reportPrivateUsage]
    _graph_entity_names,  # pyright: ignore[reportPrivateUsage]
    _MemoryEvidence,  # pyright: ignore[reportPrivateUsage]
    _MemoryFact,  # pyright: ignore[reportPrivateUsage]
    _MemoryRelation,  # pyright: ignore[reportPrivateUsage]
    _prioritize_memory_evidence,  # pyright: ignore[reportPrivateUsage]
    _recall_context,  # pyright: ignore[reportPrivateUsage]
    _relation_extraction_input,  # pyright: ignore[reportPrivateUsage]
    _relation_record_id,  # pyright: ignore[reportPrivateUsage]
    _remember_context,  # pyright: ignore[reportPrivateUsage]
)
from draive.surreal.types import SurrealObject


def test_classify_entity_operations_remembers_unknown_entity_as_addition() -> None:
    operation = EntityOperation(
        operation="remember",
        name="Alice",
        summary="Likes tea.",
        retention="long_term",
    )

    deletions, updates, additions, migrated = _classify_entity_operations([operation], {})

    assert deletions == []
    assert updates == []
    assert additions == [operation]
    assert migrated == {}


def test_classify_entity_operations_remembers_known_entity_as_update() -> None:
    node_ids: Mapping[tuple[MemoryLayer, str], str] = {("long_term", "Alice"): "id-1"}
    operation = EntityOperation(
        operation="remember",
        name="Alice",
        summary="Likes coffee now.",
        retention="long_term",
    )

    deletions, updates, additions, migrated = _classify_entity_operations([operation], node_ids)

    assert deletions == []
    assert additions == []
    assert updates == [("long_term", "id-1", operation)]
    assert migrated == {}


def test_classify_entity_operations_migrates_layer_on_retention_change() -> None:
    # Entity was first recorded as short-term, but the model now judges it a durable,
    # long-term fact - it must move to the "long_term" graph rather than being
    # updated in place in "short_term", or it would be lost once the thread/session ends.
    node_ids: Mapping[tuple[MemoryLayer, str], str] = {("short_term", "Alice"): "id-1"}
    operation = EntityOperation(
        operation="remember",
        name="Alice",
        summary="Uses a wheelchair.",
        retention="long_term",
    )

    deletions, updates, additions, migrated = _classify_entity_operations([operation], node_ids)

    assert deletions == [("short_term", "id-1", "Alice")]
    assert updates == []
    assert additions == [operation]
    assert migrated == {"Alice": ("short_term", "id-1", "long_term")}


def test_classify_entity_operations_forget_unknown_entity_is_noop() -> None:
    operation = EntityOperation(operation="forget", name="Ghost")

    deletions, updates, additions, migrated = _classify_entity_operations([operation], {})

    assert deletions == []
    assert updates == []
    assert additions == []
    assert migrated == {}


def test_classify_entity_operations_forgets_all_matching_entities() -> None:
    node_ids: Mapping[tuple[MemoryLayer, str], str] = {
        ("long_term", "Alice"): "agent-id",
        ("mid_term", "Alice"): "thread-id",
        ("short_term", "Alice"): "agent-thread-id",
        ("long_term", "Bob"): "bob-id",
    }
    operation = EntityOperation(operation="forget", name="Alice")

    deletions, updates, additions, migrated = _classify_entity_operations([operation], node_ids)

    assert deletions == [
        ("long_term", "agent-id", "Alice"),
        ("mid_term", "thread-id", "Alice"),
        ("short_term", "agent-thread-id", "Alice"),
    ]
    assert updates == []
    assert additions == []
    assert migrated == {}


def test_classify_entity_operations_remember_wins_over_forget_for_same_entity() -> None:
    # Regression test: a model that (despite being told not to) pairs a "forget" and a
    # "remember" for the same entity must not have the correction silently swallowed by
    # the deletion - the remember has to win.
    node_ids: Mapping[tuple[MemoryLayer, str], str] = {("long_term", "Alice"): "id-1"}
    forget = EntityOperation(operation="forget", name="Alice")
    remember = EntityOperation(
        operation="remember",
        name="Alice",
        summary="Corrected fact.",
        retention="long_term",
    )

    deletions, updates, additions, migrated = _classify_entity_operations(
        [forget, remember], node_ids
    )

    assert deletions == []
    assert updates == [("long_term", "id-1", remember)]
    assert additions == []
    assert migrated == {}


def test_classify_entity_operations_keeps_only_latest_remember_per_name() -> None:
    # Regression test: two "remember" operations for the same name cannot both be
    # honored - they may disagree about retention, previously producing duplicate
    # forget entries (a `KeyError` in `_apply_entity_deletions`) or an entity duplicated
    # across two graphs. Classification keeps only the latest remember per name.
    node_ids: Mapping[tuple[MemoryLayer, str], str] = {("mid_term", "Bob"): "id-1"}
    first = EntityOperation(
        operation="remember", name="Bob", summary="Fact A.", retention="short_term"
    )
    second = EntityOperation(
        operation="remember", name="Bob", summary="Fact B.", retention="long_term"
    )

    deletions, updates, additions, migrated = _classify_entity_operations([first, second], node_ids)

    assert deletions == [("mid_term", "id-1", "Bob")]
    assert updates == []
    assert additions == [second]  # the earlier, superseded remember is discarded entirely
    assert migrated == {"Bob": ("mid_term", "id-1", "long_term")}


def test_classify_entity_operations_duplicate_remember_never_updates_deleted_record() -> None:
    # Regression test: when one duplicate "remember" resolves to an in-place update and
    # the other to a migration, letting both through queued an `UPDATE` against the very
    # record the migration deletes - the update silently no-oped and its summary was
    # lost. Last-wins dedupe makes the outcome deterministic in both orders.
    node_ids: Mapping[tuple[MemoryLayer, str], str] = {("mid_term", "Bob"): "id-1"}
    update_op = EntityOperation(
        operation="remember", name="Bob", summary="Fact A.", retention="mid_term"
    )
    migrate_op = EntityOperation(
        operation="remember", name="Bob", summary="Fact B.", retention="long_term"
    )

    # migration is the latest judgment - no stale update against the deleted record
    deletions, updates, additions, migrated = _classify_entity_operations(
        [update_op, migrate_op], node_ids
    )
    assert deletions == [("mid_term", "id-1", "Bob")]
    assert updates == []
    assert additions == [migrate_op]
    assert migrated == {"Bob": ("mid_term", "id-1", "long_term")}

    # the in-layer update is the latest judgment - no migration happens at all
    deletions, updates, additions, migrated = _classify_entity_operations(
        [migrate_op, update_op], node_ids
    )
    assert deletions == []
    assert updates == [("mid_term", "id-1", update_op)]
    assert additions == []
    assert migrated == {}


def test_relation_extraction_input_labels_layer_and_sorts_names() -> None:
    rendered = _relation_extraction_input("user: hi", "long_term", {"Bob", "Alice"})

    # one name per line (names may contain commas), sorted, inside the <entities> block
    assert "<entities>\nAlice\nBob\n</entities>" in rendered
    assert "<exchange>\nuser: hi\n</exchange>" in rendered
    assert surreal_agent_memory._LAYER_LABELS["long_term"] in rendered  # pyright: ignore[reportPrivateUsage]


def test_prioritize_memory_evidence_keeps_distinct_layers_for_shared_names() -> None:
    # Entity names are only unique within a single graph - a name appearing in
    # multiple layers may be the same real-world entity remembered redundantly, or
    # two entirely distinct entities that happen to share a label (see the
    # preferred-layer comment in `_classify_entity_operations`). Either way, the
    # code must never silently delete a layer's fact or relation purely because
    # another layer surfaced a same-named record first - that call is left to
    # `_RECALL_SEARCH_INSTRUCTIONS`, which sees every layer's evidence.
    prioritized = _prioritize_memory_evidence(
        {
            "long_term": _MemoryEvidence(
                facts=(
                    _MemoryFact(
                        identifier="agent-sam",
                        name="Sam",
                        summary="Long-term Sam summary.",
                    ),
                    _MemoryFact(
                        identifier="agent-alex",
                        name="Alex",
                        summary="Long-term Alex summary.",
                    ),
                ),
                relations=(
                    _MemoryRelation(
                        source_identifier="agent-sam",
                        source_name="Sam",
                        source_summary="Long-term Sam summary.",
                        target_identifier="agent-alex",
                        target_name="Alex",
                        target_summary="Long-term Alex summary.",
                        label="reports to",
                    ),
                ),
            ),
            "short_term": _MemoryEvidence(
                facts=(
                    _MemoryFact(
                        identifier="local-sam",
                        name="Sam",
                        summary="Local thread Sam summary.",
                    ),
                    _MemoryFact(
                        identifier="local-alex",
                        name="Alex",
                        summary="Local thread Alex summary.",
                    ),
                ),
                relations=(
                    _MemoryRelation(
                        source_identifier="local-sam",
                        source_name="Sam",
                        source_summary="Local thread Sam summary.",
                        target_identifier="local-alex",
                        target_name="Alex",
                        target_summary="Local thread Alex summary.",
                        label="reports to",
                    ),
                ),
            ),
        }
    )

    # both layers survive, in recall-priority order - neither one's fact or relation
    # is dropped just because the other layer has a same-named record
    assert tuple(prioritized) == ("short_term", "long_term")
    assert prioritized["short_term"].facts == (
        _MemoryFact(identifier="local-sam", name="Sam", summary="Local thread Sam summary."),
        _MemoryFact(identifier="local-alex", name="Alex", summary="Local thread Alex summary."),
    )
    assert prioritized["short_term"].relations == (
        _MemoryRelation(
            source_identifier="local-sam",
            source_name="Sam",
            source_summary="Local thread Sam summary.",
            target_identifier="local-alex",
            target_name="Alex",
            target_summary="Local thread Alex summary.",
            label="reports to",
        ),
    )
    assert prioritized["long_term"].facts == (
        _MemoryFact(identifier="agent-sam", name="Sam", summary="Long-term Sam summary."),
        _MemoryFact(identifier="agent-alex", name="Alex", summary="Long-term Alex summary."),
    )
    assert prioritized["long_term"].relations == (
        _MemoryRelation(
            source_identifier="agent-sam",
            source_name="Sam",
            source_summary="Long-term Sam summary.",
            target_identifier="agent-alex",
            target_name="Alex",
            target_summary="Long-term Alex summary.",
            label="reports to",
        ),
    )


def test_prioritize_memory_evidence_dedupes_only_exact_repeated_records() -> None:
    prioritized = _prioritize_memory_evidence(
        {
            "short_term": _MemoryEvidence(
                facts=(
                    _MemoryFact(identifier="local-sam-1", name="Sam", summary="Sam summary."),
                    _MemoryFact(identifier="local-sam-1", name="Sam", summary="Sam summary."),
                    _MemoryFact(identifier="local-sam-2", name="Sam", summary="Duplicate Sam."),
                    _MemoryFact(identifier="local-alex", name="Alex", summary="Alex summary."),
                ),
                relations=(
                    _MemoryRelation(
                        source_identifier="local-sam-2",
                        source_name="Sam",
                        source_summary="Duplicate Sam.",
                        target_identifier="local-alex",
                        target_name="Alex",
                        target_summary="Alex summary.",
                        label="reports to",
                    ),
                ),
            ),
        }
    )

    # "local-sam-1" and "local-sam-2" are distinct records sharing a name - both are
    # kept; only the exact repeated "local-sam-1" record is collapsed.
    assert prioritized["short_term"].facts == (
        _MemoryFact(identifier="local-sam-1", name="Sam", summary="Sam summary."),
        _MemoryFact(identifier="local-sam-2", name="Sam", summary="Duplicate Sam."),
        _MemoryFact(identifier="local-alex", name="Alex", summary="Alex summary."),
    )
    assert prioritized["short_term"].relations == (
        _MemoryRelation(
            source_identifier="local-sam-2",
            source_name="Sam",
            source_summary="Duplicate Sam.",
            target_identifier="local-alex",
            target_name="Alex",
            target_summary="Alex summary.",
            label="reports to",
        ),
    )


def _patch_recording_surreal(
    monkeypatch: pytest.MonkeyPatch,
    /,
) -> tuple[list[Any], list[Any]]:
    execute_calls: list[Any] = []
    relate_calls: list[Any] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        execute_calls.append((statement, variables))
        return ()  # no existing relation found

    async def fake_relate(*args: Any, **kwargs: Any) -> str | None:
        relate_calls.append((args, kwargs))
        return "long_term_memory_relation:1"

    monkeypatch.setattr(surreal_agent_memory.Surreal, "execute", fake_execute)
    monkeypatch.setattr(surreal_agent_memory.Surreal, "relate", fake_relate)

    return execute_calls, relate_calls


@pytest.mark.asyncio
async def test_apply_relation_operations_drops_names_outside_layer_without_touching_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute_calls, relate_calls = _patch_recording_surreal(monkeypatch)

    node_ids: Mapping[tuple[MemoryLayer, str], str] = {("long_term", "Alice"): "a1"}
    relation = RelationOperation(operation="add", source="Alice", target="Bob", label="knows")

    async with ctx.scope("test.surreal.agent_memory.drop_relation"):
        await _apply_relation_operations([relation], layer="long_term", node_ids=node_ids)

    # unknown target name -> dropped before any DB interaction
    assert execute_calls == []
    assert relate_calls == []


@pytest.mark.asyncio
async def test_apply_relation_operations_creates_relation_between_known_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execute_calls, relate_calls = _patch_recording_surreal(monkeypatch)

    node_ids: Mapping[tuple[MemoryLayer, str], str] = {
        ("long_term", "Alice"): "a1",
        ("long_term", "Bob"): "b1",
    }
    relation = RelationOperation(operation="add", source="Alice", target="Bob", label="knows")

    async with ctx.scope("test.surreal.agent_memory.create_relation"):
        await _apply_relation_operations([relation], layer="long_term", node_ids=node_ids)

    # relations are created with a deterministic edge id via RELATE, not `Surreal.relate`
    assert relate_calls == []
    relate_statements = [
        (statement, variables)
        for statement, variables in execute_calls
        if statement.startswith("RELATE ")
    ]
    assert len(relate_statements) == 1
    statement, variables = relate_statements[0]
    expected_identifier = _relation_record_id(
        RecordID("long_term_memory_node", "a1"),
        RecordID("long_term_memory_node", "b1"),
        "knows",
    )
    assert f"long_term_memory_relation:⟨{expected_identifier}⟩" in statement
    assert variables["_source"] == RecordID("long_term_memory_node", "a1")
    assert variables["_target"] == RecordID("long_term_memory_node", "b1")
    assert variables["_label"] == "knows"


@pytest.mark.asyncio
async def test_graph_entity_names_maps_name_to_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        assert "SELECT id, name, updated FROM long_term_memory_node" in statement
        assert variables["scope_key"] == "agent://assistant"
        return (
            {"id": "p1", "name": "Priya"},
            {"id": "t1", "name": "Tom"},
        )

    monkeypatch.setattr(surreal_agent_memory.Surreal, "execute", fake_execute)

    async with ctx.scope("test.surreal.agent_memory.scope_entity_names"):
        names = await _graph_entity_names(
            table="long_term_memory_node",
            scope_key="agent://assistant",
            limit=500,
        )

    assert dict(names) == {"Priya": "p1", "Tom": "t1"}


def _patch_noop_vector_index(monkeypatch: pytest.MonkeyPatch, /) -> None:
    # Recall's `find_memory` ensures HNSW indexes (sequential DDL) before its gathered
    # searches - a no-op stand-in keeps tests that fake `_search_graph` away from the DB.
    async def fake_ensure_vector_index(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(surreal_agent_memory, "_ensure_vector_index", fake_ensure_vector_index)


def _fake_embedding(vector: Sequence[float]) -> TextEmbedding:
    async def embedding(
        values: Sequence[str],
        /,
        attribute: Any = None,
        **extra: Any,
    ) -> Sequence[Embedded[str]]:
        return [Embedded(value=value, vector=vector) for value in values]

    return TextEmbedding(embedding=embedding)


@pytest.mark.asyncio
async def test_remember_context_links_relation_to_entity_missed_by_similarity_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a live-fire finding: a relation to an already-known entity must
    still be created even when that entity does not surface via the similarity-gated
    existing-entity lookup for this turn (e.g. a first-person message that never names the
    person a new fact is about). As long as the entity exists somewhere in the graph,
    `_graph_entity_names` must make it resolvable as a relation endpoint - `node_ids` alone
    (populated only from whatever the KNN existing-entity search retrieved) is not enough.
    """

    async def fake_search_graph(**kwargs: Any) -> Sequence[_MemoryFact]:
        return ()  # nothing surfaces via similarity this turn, for any layer

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)

    execute_calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_execute(
        statement: str,
        /,
        **variables: Any,
    ) -> Sequence[SurrealObject]:
        execute_calls.append((statement, variables))
        if statement.strip().startswith("SELECT id, name, updated FROM"):
            if variables.get("scope_key") == "agent://assistant":
                return ({"id": "p1", "name": "Priya"},)
            return ()
        return ()  # DEFINE INDEX / UPSERT / relation-exists lookup - none matter here

    monkeypatch.setattr(surreal_agent_memory.Surreal, "execute", fake_execute)

    async def fake_generating(
        generated: type[Any],
        /,
        *,
        instructions: Any,
        input: Any,  # noqa: A002
        toolbox: Any,
        examples: Any,
        decoder: Any,
        **extra: Any,
    ) -> Any:
        if generated is EntityExtraction:
            return EntityExtraction(
                entities=(
                    EntityOperation(
                        operation="remember",
                        name="LH123456",
                        summary="Priya's frequent flyer number.",
                        retention="long_term",
                    ),
                )
            )

        if generated is RelationExtraction:
            return RelationExtraction(
                relations=(
                    RelationOperation(
                        operation="add",
                        source="Priya",
                        target="LH123456",
                        label="has frequent flyer number",
                    ),
                )
            )

        raise AssertionError(f"Unexpected generated type: {generated}")

    thread = AgentThread.of(uuid4())
    context = (
        ModelInput.of(MultimodalContent.of("My frequent flyer number is LH123456.")),
        ModelOutput.of(MultimodalContent.of("Noted.")),
    )

    async with ctx.scope(
        "test.surreal.agent_memory.relation_reaches_missed_entity",
        _fake_embedding([0.1]),
        ModelGeneration(generating=fake_generating),
    ):
        await _remember_context(
            agent_uri="agent://assistant",
            thread=thread,
            context=context,
            search_effort=40,
            score_threshold=0.5,
            existing_lookup_limit=8,
            max_nodes_per_scope=None,
            memory_guidelines=None,
        )

    relate_statements = [
        (statement, variables)
        for statement, variables in execute_calls
        if statement.startswith("RELATE ")
    ]
    assert len(relate_statements) == 1
    _, variables = relate_statements[0]
    # Priya's identifier, resolved despite missing the similarity hit
    assert variables["_source"] == RecordID("long_term_memory_node", "p1")
    assert variables["_target"] == RecordID(
        "long_term_memory_node",
        _entity_record_id("agent://assistant", "LH123456"),
    )
    assert variables["_label"] == "has frequent flyer number"


@pytest.mark.asyncio
async def test_remember_context_appends_plain_string_memory_guidelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_graph(**kwargs: Any) -> Sequence[_MemoryFact]:
        return ()

    async def fake_execute(statement: str, /, **variables: Any) -> Sequence[SurrealObject]:
        return ()  # no existing entities in any layer - relation gathering finds nothing

    captured: dict[str, Any] = {}

    async def fake_generating(
        generated: type[Any],
        /,
        *,
        instructions: Any,
        **extra: Any,
    ) -> Any:
        assert generated is EntityExtraction
        captured["instructions"] = instructions
        return EntityExtraction(entities=())

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)
    monkeypatch.setattr(surreal_agent_memory.Surreal, "execute", fake_execute)

    thread = AgentThread.of(uuid4())
    context = (ModelInput.of(MultimodalContent.of("My card number is 4242 4242 4242 4242.")),)

    async with ctx.scope(
        "test.surreal.agent_memory.remember_guidelines_plain",
        _fake_embedding([0.1]),
        ModelGeneration(generating=fake_generating),
    ):
        await _remember_context(
            agent_uri="agent://assistant",
            thread=thread,
            context=context,
            search_effort=40,
            score_threshold=None,
            existing_lookup_limit=8,
            max_nodes_per_scope=None,
            memory_guidelines="Never store full payment card numbers; redact all but last 4.",
        )

    instructions = captured["instructions"]
    assert "Never store full payment card numbers; redact all but last 4." in instructions
    assert "Apply them especially to long-term agent-specific knowledge" in instructions
    # base instructions must precede the appended guidelines section
    assert "layered knowledge graph memory" in captured["instructions"]
    assert captured["instructions"].index("layered knowledge graph memory") < captured[
        "instructions"
    ].index("Never store full payment card numbers")


@pytest.mark.asyncio
async def test_remember_context_resolves_template_memory_guidelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_graph(**kwargs: Any) -> Sequence[_MemoryFact]:
        return ()

    async def fake_execute(statement: str, /, **variables: Any) -> Sequence[SurrealObject]:
        return ()  # no existing entities in any layer - relation gathering finds nothing

    captured: dict[str, Any] = {}

    async def fake_generating(
        generated: type[Any],
        /,
        *,
        instructions: Any,
        **extra: Any,
    ) -> Any:
        assert generated is EntityExtraction
        captured["instructions"] = instructions
        return EntityExtraction(entities=())

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)
    monkeypatch.setattr(surreal_agent_memory.Surreal, "execute", fake_execute)

    thread = AgentThread.of(uuid4())
    context = (ModelInput.of(MultimodalContent.of("I just adopted a cat named Momo.")),)

    async with ctx.scope(
        "test.surreal.agent_memory.remember_guidelines_template",
        _fake_embedding([0.1]),
        ModelGeneration(generating=fake_generating),
        TemplatesRepository.volatile(
            pet_guidelines="Always record pet names and species as long_term facts."
        ),
    ):
        await _remember_context(
            agent_uri="agent://assistant",
            thread=thread,
            context=context,
            search_effort=40,
            score_threshold=None,
            existing_lookup_limit=8,
            max_nodes_per_scope=None,
            memory_guidelines=Template.of("pet_guidelines"),
        )

    assert "Always record pet names and species as long_term facts." in captured["instructions"]


@pytest.mark.asyncio
async def test_remember_context_without_memory_guidelines_omits_extra_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_graph(**kwargs: Any) -> Sequence[_MemoryFact]:
        return ()

    async def fake_execute(statement: str, /, **variables: Any) -> Sequence[SurrealObject]:
        return ()  # no existing entities in any layer - relation gathering finds nothing

    captured: dict[str, Any] = {}

    async def fake_generating(
        generated: type[Any],
        /,
        *,
        instructions: Any,
        **extra: Any,
    ) -> Any:
        assert generated is EntityExtraction
        captured["instructions"] = instructions
        return EntityExtraction(entities=())

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)
    monkeypatch.setattr(surreal_agent_memory.Surreal, "execute", fake_execute)

    thread = AgentThread.of(uuid4())
    context = (ModelInput.of(MultimodalContent.of("Hello there.")),)

    async with ctx.scope(
        "test.surreal.agent_memory.remember_guidelines_none",
        _fake_embedding([0.1]),
        ModelGeneration(generating=fake_generating),
    ):
        await _remember_context(
            agent_uri="agent://assistant",
            thread=thread,
            context=context,
            search_effort=40,
            score_threshold=None,
            existing_lookup_limit=8,
            max_nodes_per_scope=None,
            memory_guidelines=None,
        )

    assert "Additional memory guidelines" not in captured["instructions"]
    assert '"remember": store a currently true fact about an entity' in captured["instructions"]
    assert '"forget": remove an existing entity by exact `name`' in captured["instructions"]
    assert '"short_term": same agent, same thread' in captured["instructions"]
    assert '"mid_term": same thread, all agents' in captured["instructions"]
    assert '"long_term": same agent, future threads' in captured["instructions"]


def _stream_of(*chunks: ModelOutputChunk) -> AsyncIterable[ModelOutputChunk]:
    async def stream() -> AsyncIterable[ModelOutputChunk]:
        for chunk in chunks:
            yield chunk

    return stream()


@pytest.mark.asyncio
async def test_recall_context_passes_through_untouched_when_message_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_search_graph(**kwargs: Any) -> Sequence[_MemoryFact]:
        calls.append(kwargs["scope_key"])
        return ()

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)

    thread = AgentThread.of(uuid4())
    model_input = ModelInput.of(MultimodalContent.empty)

    async with ctx.scope("test.surreal.agent_memory.recall_empty", _fake_embedding([0.1])):
        result = await _recall_context(
            agent_uri="agent://assistant",
            thread=thread,
            input=model_input,
            search_limit=5,
            search_effort=40,
            score_threshold=None,
        )

    assert result == (model_input,)
    assert calls == []  # no embedding/search work done for empty content


@pytest.mark.asyncio
async def test_recall_context_always_searches_before_answering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_calls: list[str] = []

    async def fake_search_graph(**kwargs: Any) -> Sequence[_MemoryFact]:
        search_calls.append(kwargs["scope_key"])
        return ()

    completion_calls = 0

    def fake_generating(
        *,
        instructions: str,
        tools: Any,
        context: Any,
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            # the toolbox forces a search on the first iteration regardless of the message
            return _stream_of(
                ModelToolRequest.of("c1", tool="find_memory", arguments={"query": "weather"})
            )

        return _stream_of(TextContent.of("What's the weather like?"))

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)

    thread = AgentThread.of(uuid4())
    model_input = ModelInput.of(MultimodalContent.of("What's the weather like?"))

    _patch_noop_vector_index(monkeypatch)

    async with ctx.scope(
        "test.surreal.agent_memory.recall_no_facts",
        _fake_embedding([0.1]),
        GenerativeModel(generating=fake_generating),
    ):
        result = await _recall_context(
            agent_uri="agent://assistant",
            thread=thread,
            input=model_input,
            search_limit=5,
            search_effort=40,
            score_threshold=None,
        )

    assert completion_calls == 2  # forced search, then a final rewrite once nothing turned up
    assert len(search_calls) == 3  # one semantic discovery call per graph layer
    assert len(result) == 1
    rephrased = result[0]
    assert isinstance(rephrased, ModelInput)
    assert rephrased.content.to_str() == "What's the weather like?"


@pytest.mark.asyncio
async def test_recall_context_rephrases_with_recalled_facts_and_preserves_other_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_graph(
        *,
        scope_key: str,
        expand_neighbors: bool = True,
        **kwargs: Any,
    ) -> Sequence[_MemoryFact]:
        if scope_key == "agent://assistant":
            return (_MemoryFact(identifier="id-1", name="Sam", summary="Works as a nurse."),)

        return ()

    captured: dict[str, Any] = {}
    completion_calls = 0

    def fake_generating(
        *,
        instructions: str,
        tools: Any,
        context: Any,
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            captured["instructions"] = instructions
            return _stream_of(
                ModelToolRequest.of("c1", tool="find_memory", arguments={"query": "Sam"})
            )

        captured["second_call_context"] = context
        return _stream_of(
            TextContent.of("What's the weather like where Sam, who works as a nurse, lives?")
        )

    async def fake_relations_for_records(**kwargs: Any) -> Sequence[Any]:
        return ()

    async def fake_touch_memory_evidence(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)
    monkeypatch.setattr(
        surreal_agent_memory,
        "_relations_for_records",
        fake_relations_for_records,
    )
    monkeypatch.setattr(
        surreal_agent_memory,
        "_touch_memory_evidence",
        fake_touch_memory_evidence,
    )

    thread = AgentThread.of(uuid4())
    artifact_part = ArtifactContent.of(EntityOperation(operation="remember", name="x", summary="y"))
    model_input = ModelInput.of(
        MultimodalContent.of(
            TextContent.of("What's the weather like where Sam lives?"),
            artifact_part,
        )
    )

    _patch_noop_vector_index(monkeypatch)

    async with ctx.scope(
        "test.surreal.agent_memory.recall_with_facts",
        _fake_embedding([0.1]),
        GenerativeModel(generating=fake_generating),
    ):
        result = await _recall_context(
            agent_uri="agent://assistant",
            thread=thread,
            input=model_input,
            search_limit=5,
            search_effort=40,
            score_threshold=None,
        )

    # the tool response fed back into the second completion call carries the recalled fact
    context_text = "".join(
        response.content.to_str()
        for element in captured["second_call_context"]
        if isinstance(element, ModelInput)
        for response in element.tool_responses
    )
    assert "Sam: Works as a nurse." in context_text
    assert "unresolved" in captured["instructions"]

    assert len(result) == 1
    rephrased = result[0]
    assert isinstance(rephrased, ModelInput)
    texts = [part for part in rephrased.content.parts if isinstance(part, TextContent)]
    assert len(texts) == 1
    assert texts[0].text == "What's the weather like where Sam, who works as a nurse, lives?"
    # non-text parts of the original message must survive the rewrite untouched
    assert artifact_part in rephrased.content.parts


@pytest.mark.asyncio
async def test_recall_context_passes_relation_aware_memory_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_search_graph(
        *,
        scope_key: str,
        **kwargs: Any,
    ) -> Sequence[_MemoryFact]:
        if scope_key == "agent://assistant":
            return (_MemoryFact(identifier="sam", name="Sam", summary="Works as a nurse."),)

        return ()

    async def fake_relations_for_records(**kwargs: Any) -> Sequence[_MemoryRelation]:
        return (
            _MemoryRelation(
                source_identifier="sam",
                source_name="Sam",
                source_summary="Works as a nurse.",
                target_identifier="alex",
                target_name="Alex",
                target_summary="Sam's manager.",
                label="reports to",
            ),
        )

    captured: dict[str, Any] = {}
    completion_calls = 0

    def fake_generating(
        *,
        context: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            return _stream_of(
                ModelToolRequest.of("c1", tool="find_memory", arguments={"query": "Sam"})
            )

        captured["second_call_context"] = context
        return _stream_of(TextContent.of("Who does Sam, who reports to Alex, report to?"))

    async def fake_touch_memory_evidence(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)
    monkeypatch.setattr(
        surreal_agent_memory,
        "_relations_for_records",
        fake_relations_for_records,
    )
    monkeypatch.setattr(
        surreal_agent_memory,
        "_touch_memory_evidence",
        fake_touch_memory_evidence,
    )

    thread = AgentThread.of(uuid4())
    model_input = ModelInput.of(MultimodalContent.of("Who does Sam report to?"))

    _patch_noop_vector_index(monkeypatch)

    async with ctx.scope(
        "test.surreal.agent_memory.recall_relation_evidence",
        _fake_embedding([0.1]),
        GenerativeModel(generating=fake_generating),
    ):
        await _recall_context(
            agent_uri="agent://assistant",
            thread=thread,
            input=model_input,
            search_limit=5,
            search_effort=40,
            score_threshold=None,
        )

    context_text = "".join(
        response.content.to_str()
        for element in captured["second_call_context"]
        if isinstance(element, ModelInput)
        for response in element.tool_responses
    )
    assert "Sam: Works as a nurse." in context_text
    assert "Alex: Sam's manager." in context_text
    assert "Sam --reports to--> Alex" in context_text


@pytest.mark.asyncio
async def test_recall_context_follows_up_with_a_second_search_memory_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_calls: list[str] = []

    async def fake_search_graph(**kwargs: Any) -> Sequence[_MemoryFact]:
        search_calls.append(kwargs["scope_key"])
        if kwargs["scope_key"] == "agent://assistant":
            return (_MemoryFact(identifier="id-1", name="Sam", summary="Reports to Alex."),)

        return ()

    completion_calls = 0

    def fake_generating(
        *,
        instructions: str,
        tools: Any,
        context: Any,
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            return _stream_of(
                ModelToolRequest.of("c1", tool="find_memory", arguments={"query": "Sam"})
            )

        if completion_calls == 2:
            # the model follows up on "Alex", surfaced by the first search, on its own
            return _stream_of(
                ModelToolRequest.of("c2", tool="find_memory", arguments={"query": "Alex"})
            )

        return _stream_of(TextContent.of("final"))

    async def fake_relations_for_records(**kwargs: Any) -> Sequence[Any]:
        return ()

    async def fake_touch_memory_evidence(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)
    monkeypatch.setattr(
        surreal_agent_memory,
        "_relations_for_records",
        fake_relations_for_records,
    )
    monkeypatch.setattr(
        surreal_agent_memory,
        "_touch_memory_evidence",
        fake_touch_memory_evidence,
    )

    thread = AgentThread.of(uuid4())
    model_input = ModelInput.of(MultimodalContent.of("Who does Sam report to?"))

    _patch_noop_vector_index(monkeypatch)

    async with ctx.scope(
        "test.surreal.agent_memory.recall_multi_hop",
        _fake_embedding([0.1]),
        GenerativeModel(generating=fake_generating),
    ):
        result = await _recall_context(
            agent_uri="agent://assistant",
            thread=thread,
            input=model_input,
            search_limit=5,
            search_effort=40,
            score_threshold=None,
        )

    # two full find_memory calls (3 graph layers each), no hard iteration cap involved -
    # the loop stops because the model itself stopped requesting tools
    assert completion_calls == 3
    assert len(search_calls) == 6
    assert len(result) == 1
    assert isinstance(result[0], ModelInput)
    assert result[0].content.to_str() == "final"


@pytest.mark.asyncio
async def test_resolve_entity_extraction_instructions_falls_back_when_template_unresolved() -> None:
    """A `memory_guidelines` `Template` that no configured `TemplatesRepository` can resolve
    must degrade to the base instructions instead of raising `TemplateMissing` and breaking
    every single `remember()` call.
    """
    async with ctx.scope("test.surreal.agent_memory.template_missing"):
        instructions = await surreal_agent_memory._resolve_entity_extraction_instructions(  # pyright: ignore[reportPrivateUsage]
            Template.of("nonexistent-guidelines")
        )

    assert instructions == surreal_agent_memory._ENTITY_EXTRACTION_INSTRUCTIONS  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_recall_context_does_not_duplicate_parts_when_rephrase_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: when the rewrite loop ends with no text content, recall must return
    the original message unchanged rather than re-appending its own non-text parts on top
    of itself, which used to duplicate e.g. images in the final `ModelInput`.
    """

    async def fake_search_graph(**kwargs: Any) -> Sequence[_MemoryFact]:
        return ()

    completion_calls = 0

    def fake_generating(
        *,
        instructions: str,
        tools: Any,
        context: Any,
        output: Any,
        **extra: Any,
    ) -> AsyncIterable[ModelOutputChunk]:
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            return _stream_of(
                ModelToolRequest.of("c1", tool="find_memory", arguments={"query": "weather"})
            )

        return _stream_of()  # the provider stops with nothing to say

    monkeypatch.setattr(surreal_agent_memory, "_search_graph", fake_search_graph)

    thread = AgentThread.of(uuid4())
    artifact_part = ArtifactContent.of(EntityOperation(operation="remember", name="x", summary="y"))
    model_input = ModelInput.of(
        MultimodalContent.of(
            TextContent.of("What's the weather like?"),
            artifact_part,
        )
    )

    _patch_noop_vector_index(monkeypatch)

    async with ctx.scope(
        "test.surreal.agent_memory.recall_empty_rephrase",
        _fake_embedding([0.1]),
        GenerativeModel(generating=fake_generating),
    ):
        result = await _recall_context(
            agent_uri="agent://assistant",
            thread=thread,
            input=model_input,
            search_limit=5,
            search_effort=40,
            score_threshold=None,
        )

    assert result == (model_input,)
    assert result[0].content.parts.count(artifact_part) == 1


def test_context_transcript_tags_media_parts_with_sender_role() -> None:
    """Regression test: media parts collected across the exchange must carry their origin -
    without this, an attachment generated by this agent and one received from its
    counterpart are indistinguishable once flattened into one list, risking misattribution
    during extraction. Roles are deliberately participant-neutral ("incoming"/"this agent"
    rather than "user"/"assistant") - this is agent memory, and the counterpart may be a
    person or another agent.
    """
    incoming_artifact = ArtifactContent.of(
        EntityOperation(operation="remember", name="u", summary="s")
    )
    own_artifact = ArtifactContent.of(EntityOperation(operation="remember", name="a", summary="s"))
    context = (
        ModelInput.of(MultimodalContent.of(incoming_artifact)),
        ModelOutput.of(MultimodalContent.of(own_artifact)),
    )

    transcript, media_parts = surreal_agent_memory._context_transcript(  # pyright: ignore[reportPrivateUsage]
        context
    )

    assert "incoming:" in transcript
    assert "this agent:" in transcript
    assert "user:" not in transcript
    assert "assistant:" not in transcript
    assert media_parts[0] == TextContent.of("[incoming attachment]")
    assert media_parts[1] == incoming_artifact
    assert media_parts[2] == TextContent.of("[this agent attachment]")
    assert media_parts[3] == own_artifact


@pytest.mark.asyncio
async def test_relations_for_records_bounds_merged_result_across_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: fetching relations per-seed at `limit` each must not let the merged,
    deduplicated result grow past `limit` overall (previously up to `limit * len(seeds)`).
    """

    async def fake_relations_for_seed(
        *,
        relation_table: str,
        seed: Any,
        limit: int,
    ) -> Sequence[SurrealObject]:
        # each seed reports `limit` distinct relations, so naive concatenation would total
        # `limit * number_of_seeds`
        return tuple(
            {
                "source": {"id": str(seed), "name": str(seed), "summary": "s"},
                "target": {
                    "id": f"target-{seed}-{i}",
                    "name": f"target-{seed}-{i}",
                    "summary": "s",
                },
                "label": "knows",
            }
            for i in range(limit)
        )

    monkeypatch.setattr(surreal_agent_memory, "_relations_for_seed", fake_relations_for_seed)

    seeds = [
        surreal_agent_memory.RecordID("node", "a"),
        surreal_agent_memory.RecordID("node", "b"),
        surreal_agent_memory.RecordID("node", "c"),
    ]

    relations = await surreal_agent_memory._relations_for_records(  # pyright: ignore[reportPrivateUsage]
        relation_table="rel",
        seeds=seeds,
        limit=5,
    )

    assert len(relations) <= 5


@pytest.mark.asyncio
async def test_remember_context_serializes_updates_sharing_a_scope_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a live-fire finding: two agents of one thread remembering
    concurrently both read-modify-write the shared mid-term graph, and the later write
    silently discarded the earlier one (a logged team decision vanished). Remember
    updates sharing any scope key must therefore run one at a time, while updates with
    fully disjoint scope keys stay concurrent.
    """
    active: int = 0
    observed_overlap: dict[str, int] = {"max": 0}

    async def fake_locked(**kwargs: Any) -> None:
        nonlocal active
        active += 1
        observed_overlap["max"] = max(observed_overlap["max"], active)
        await asyncio.sleep(0.001)
        active -= 1

    monkeypatch.setattr(surreal_agent_memory, "_remember_context_locked", fake_locked)

    context = (ModelInput.of(MultimodalContent.of("hello")),)
    shared_thread = AgentThread.of(uuid4())

    async def remember(agent_uri: str, thread: AgentThread) -> None:
        await _remember_context(
            agent_uri=agent_uri,
            thread=thread,
            context=context,
            search_effort=40,
            score_threshold=None,
            existing_lookup_limit=8,
            max_nodes_per_scope=None,
            memory_guidelines=None,
        )

    # Two agents in the same thread share the mid-term scope key - never overlapping.
    await asyncio.gather(
        remember("agent://one", shared_thread),
        remember("agent://two", shared_thread),
    )
    assert observed_overlap["max"] == 1

    # Different agents in different threads share nothing - free to overlap.
    observed_overlap["max"] = 0
    await asyncio.gather(
        remember("agent://one", AgentThread.of(uuid4())),
        remember("agent://two", AgentThread.of(uuid4())),
    )
    assert observed_overlap["max"] == 2
