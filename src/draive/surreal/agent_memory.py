import asyncio
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AsyncExitStack
from typing import Any, Literal, NoReturn, cast, final
from uuid import UUID
from weakref import WeakValueDictionary

from haiway import Meta, MetaValues, State, ctx, retry
from surrealdb import RecordID

from draive.agents import AgentIdentity, AgentMemory, AgentThread
from draive.embedding import Embedded, TextEmbedding
from draive.generation import ModelGeneration
from draive.models import ModelContext, ModelInput, ModelInstructions
from draive.multimodal import (
    MultimodalContent,
    MultimodalContentPart,
    Template,
    TemplatesRepository,
    TextContent,
)
from draive.steps import Step
from draive.surreal.state import Surreal
from draive.surreal.types import SurrealException, SurrealID, SurrealObject, SurrealValue
from draive.tools import Toolbox, tool

__all__ = ("SurrealAgentMemory",)

type MemoryLayer = Literal["short_term", "mid_term", "long_term"]
type MemoryOperation = Literal["remember", "forget"]

# Storage iteration order, most durable first; recall priority is the reverse - most
# specific first, matching the layer precedence in `_RECALL_SEARCH_INSTRUCTIONS`.
_LAYERS: tuple[MemoryLayer, ...] = ("long_term", "mid_term", "short_term")
_RECALL_LAYERS: tuple[MemoryLayer, ...] = ("short_term", "mid_term", "long_term")

# a relation always needs two distinct endpoints - skip the relation extraction pass
# entirely when fewer entities than this are known, saving an otherwise pointless call
_MIN_ENTITIES_FOR_RELATIONS: int = 2

# Cap on how many of a migrating entity's existing relations get captured for carry-over
# to its destination graph - deliberately its own constant rather than reusing
# `existing_lookup_limit` (which bounds an unrelated thing: how many existing nodes the
# extraction model sees as merge candidates), so tuning one doesn't silently tune the other.
_MAX_MIGRATED_RELATIONS_PER_ENTITY: int = 50

# Cap on how many entity names get fetched per graph as the relation-extraction closed
# list (and relation endpoint-resolution map) when capacity management is disabled
# (`max_nodes_per_scope=None`) - without it, an unbounded graph would grow the relation
# extraction prompt without limit. Capacity-managed graphs use their configured
# `max_nodes_per_scope` as the bound instead, so they are never meaningfully truncated;
# when this cap does bite, the most recently touched entities win and a warning is logged.
_MAX_RELATION_ENTITY_NAMES: int = 500

_NODE_TABLE: Mapping[MemoryLayer, str] = {
    "long_term": "long_term_memory_node",
    "mid_term": "mid_term_memory_node",
    "short_term": "short_term_memory_node",
}

_RELATION_TABLE: Mapping[MemoryLayer, str] = {
    "long_term": "long_term_memory_relation",
    "mid_term": "mid_term_memory_relation",
    "short_term": "short_term_memory_relation",
}

# Model-facing layer descriptions - shown in extraction inputs (grouping existing
# memory), relation inputs ("Memory layer: ..."), and recall tool evidence.
_LAYER_LABELS: Mapping[MemoryLayer, str] = {
    "long_term": "Long-term memory (this agent, across all conversations)",
    "mid_term": "Mid-term memory (this conversation, shared by all its agents)",
    "short_term": "Short-term memory (this agent, this conversation only)",
}

_ENTITY_EXTRACTION_INSTRUCTIONS: ModelInstructions = """\
You maintain a layered knowledge graph memory for an agent. Turn the latest exchange
into "remember"/"forget" entity operations, or return no operations when memory should
not change.

<input_format>
- <exchange>: transcript of the latest exchange. "incoming" lines arrived from the
  agent's counterpart in this thread - a person or another agent; "this agent" lines
  are the agent's own replies. Attachments may follow, tagged with their origin.
- <existing_memory>: entities already stored, grouped by memory layer, one per line as
  `- "name": summary`. The quoted part is that entity's exact name.
</input_format>

<operations>
- "remember": store a currently true fact about an entity, identified by a stable
  real-world `name`. This covers both new facts and corrections: remembering an already
  known name replaces its old summary in place instead of creating a duplicate, so a
  correction is expressed as "remember" alone with the updated summary. Never pair
  "remember" of a corrected fact with a "forget" of the same entity.
- "forget": remove an existing entity by exact `name`, only when nothing should replace
  it - it is false, obsolete, or no longer relevant, and the exchange gives no corrected
  fact to store in its place. If a corrected fact exists, that is a "remember", not a
  "forget".
</operations>

<entity_names>
- `name` holds only the entity's bare name: never a whole `"name": summary` line copied
  from <existing_memory>, never surrounding quotes, never the summary.
- Use real entity names, never conversation-role placeholders like "user", "assistant",
  "incoming", "this agent", "I", or "you".
- Once an entity exists in memory, always reuse its exact name, character-for-character.
  Never introduce a reworded, expanded, or more specific variant of an existing name -
  that creates a disconnected duplicate instead of updating the original.
- Before adding a new entity, check whether the fact is just another detail of an entity
  already in <existing_memory>: a deadline, status, attribute, or update for something
  that already has a node is an update to that entity's summary, not a new entity. Never
  create narrow spin-offs like "Project X deadline" or "Project X backend status"
  alongside an existing "Project X" - fold the deadline and the status into "Project X"
  itself. Only introduce a new entity for a distinct real-world thing with no existing
  node; prefer fewer, broader entities with rich summaries over many narrow,
  overlapping ones.
</entity_names>

<remember_fields>
`summary` - one or two sentences with all currently true relevant details, carrying
over still-valid details from the old summary when updating an existing entity:
- State only what is currently true. When a value was corrected, the old value simply
  disappears with the update - never keep superseded values, correction history, or
  "previously X" notes.
- Store facts, not payloads. Never copy bulk or processed content - documents, code,
  tool output, tables, intermediate results, or long verbatim quotes. Record what the
  thing is, where it lives, and its current status; leave out anything that can be
  regenerated or re-fetched from the source the summary names.
- Keep only the few details that would actually change how a future turn is handled.

`retention` - exactly one layer:
- "long_term": same agent, future threads. Durable facts that stay true or relevant
  beyond this conversation - identities, preferences, standing conditions, roles,
  ongoing relationships of the people, agents, and systems this agent works with, or
  lasting properties of its task domain. The identities of the work's principals -
  the client or customer, the project and its codename, key people and their roles -
  are durable in this sense even when first mentioned in passing mid-thread.
- "mid_term": same thread, all agents. Facts specific to what is happening in this
  thread that any participating agent needs to see - a decision, a scheduled event, a
  task status, or anything explicitly meant to be logged, noted, or tracked for this
  conversation.
- "short_term": same agent, same thread. This agent's own private working context -
  local focus, unresolved local reasoning, temporary decisions - describing its own
  process rather than the subject matter, needed by no other agent in the thread.

Decide retention in this order: if the fact would still be true and useful in a
completely different, future thread with this agent, choose "long_term" even if it also
matters here. Otherwise, if any agent in this thread might need it, choose "mid_term".
Otherwise choose "short_term". Never duplicate a fact across layers, and never promote
ordinary turn details or transient requests into mid/long term.
</remember_fields>

<recording_rules>
- Only information present or clearly implied in the exchange. Return no operations
  when memory should not change.
- Record facts about the world, never the conversation's own state of knowledge.
  This agent's inability to answer, its uncertainty, or its intent to look something
  up ("the client is not specified", "needs to be confirmed", "can check the
  tracker") is not a fact - never store it as a new entity and never fold it into an
  existing summary. When an exchange contains nothing but such meta-content, return
  no operations.
- A fact the incoming message clearly and directly states must be recorded even when
  this agent's own reply hedges, asks follow-up questions, or says it needs
  confirmation before storing something. The reply's uncertainty about secondary
  details (an exact date, an id, an owner, a time zone) is not the same as the core
  fact being unclear: record the core fact now, note the unresolved secondary details
  in its summary, and update that same summary once they are confirmed. Skip recording
  only when the core fact itself - not merely a secondary detail of it - is genuinely
  unclear, unstated, or in dispute.
</recording_rules>\
"""

_RELATION_EXTRACTION_INSTRUCTIONS: ModelInstructions = """\
You maintain a layered knowledge graph memory for an agent. Decide which relation
operations keep the memory's entities connected after the latest exchange, or return no
operations when nothing about relations should change.

<input_format>
- <exchange>: transcript of the latest exchange. "incoming" lines arrived from the
  agent's counterpart in this thread - a person or another agent; "this agent" lines
  are the agent's own replies. Attachments may follow, tagged with their origin.
- <entities>: the closed list of entity names that currently exist in this memory
  layer, one name per line - some existed before this exchange, some were just
  remembered because of it.
</input_format>

<operations>
- "add": a new or changed relationship between two entities, described by a short
  `label` (e.g. "works at", "prefers", "located in").
- "delete": an existing relationship that no longer holds, for example because one of
  its entities was deleted or the relationship itself changed.
</operations>

<endpoints>
`source` and `target` must each be exactly one line from <entities>,
character-for-character. Never invent a name, never use a name that is not in the list,
and never use conversation-role placeholders ("user", "assistant", "incoming",
"this agent", "I", "you", or similar) as an endpoint.
</endpoints>

<recording_rules>
Only relations actually present or clearly implied in the exchange. Return no
operations at all when nothing about relations should change, or when <entities> does
not contain enough entities to form one.
</recording_rules>\
"""

_RECALL_SEARCH_INSTRUCTIONS: ModelInstructions = """\
You rewrite an incoming message into a single, self-contained request for a downstream
language model that has no access to prior conversation history or memory.

<memory_layers>
Three retention layers, ordered by recall priority:
1. Short-term: specific to this agent within this thread. Treat this graph as the
   thread's continuation state, replacing the need to keep old messages in context.
2. Mid-term: shared by every agent in this conversation thread.
3. Long-term: kept by this agent across all conversations.
</memory_layers>

<tools>
- `find_memory`: semantic discovery across all memory layers. Use it to resolve fuzzy
  references, recover thread-continuation state, or discover candidate entities.
- `inspect_memory`: exact named-entity inspection across all memory layers. Use it
  after discovering or recognizing an entity name when you need its directly connected
  graph facts.
</tools>

<process>
1. Identify what the message actually depends on to be understood or acted on
   correctly - people, things, places, prior statements, or thread-continuation state
   it refers to without spelling out.
2. Query memory deliberately: `find_memory` for discovery and continuation state, then
   `inspect_memory` for named entities or relation follow-ups when the graph evidence
   is incomplete. Do not treat a nearby vector match as relevant unless its entity,
   summary, or relation actually helps contextualize the message.
3. Follow graph relations only while they clarify the current message. Include all
   associated facts needed to make the message standalone, but do not dump unrelated
   memory.
4. When the same entity or relation appears in multiple layers, use only the
   highest-priority layer (short-term, then mid-term, then long-term). Do not blend
   lower-priority duplicates into the rewrite.
5. For anything the message depends on that no search turned up, say so explicitly and
   naturally in the rewritten message. An unresolved dependency must stay visibly
   unresolved - never silently drop the reference or guess/invent a value for it.
</process>

<output_rules>
Respond with only the rewritten message and nothing else:
- Preserve the incoming message's intent and requested action exactly - the rewrite
  must ask for the same thing as the original, only phrased as a standalone request
  extended with the relevant recalled detail.
- Weave in only the detail actually relevant to this message - never dump every
  recalled fact in regardless of relevance, and never attach a fact or relation to the
  wrong referent.
- Do not answer the message, do not invent information beyond what was recalled, and
  do not mention that memory or search was used.
</output_rules>\
"""


@final
class EntityOperation(State, serializable=True):
    """Entity operation proposed for an agent memory graph update.

    Attributes
    ----------
    operation : MemoryOperation
        Operation to apply to the entity.
    name : str
        Canonical entity name used to match existing memory nodes.
    summary : str, default=""
        Replacement summary for remembered entities.
    retention : MemoryLayer, default="short_term"
        Retention layer selecting the memory graph for the operation.
    """

    operation: MemoryOperation
    name: str
    summary: str = ""
    retention: MemoryLayer = "short_term"


@final
class RelationOperation(State, serializable=True):
    """Relation operation proposed for an agent memory graph update.

    Attributes
    ----------
    operation : Literal["add", "delete"]
        Operation to apply to the relation.
    source : str
        Canonical name of the source entity.
    target : str
        Canonical name of the target entity.
    label : str, default=""
        Relationship label, or an empty value when deleting all matching edges.
    """

    operation: Literal["add", "delete"]
    source: str
    target: str
    label: str = ""


@final
class EntityExtraction(State, serializable=True):
    """Structured entity update returned by model generation.

    Attributes
    ----------
    entities : Sequence[EntityOperation], default=()
        Entity operations to apply.
    """

    entities: Sequence[EntityOperation] = ()


@final
class RelationExtraction(State, serializable=True):
    """Structured relation update returned by model generation.

    Attributes
    ----------
    relations : Sequence[RelationOperation], default=()
        Relation operations to apply.
    """

    relations: Sequence[RelationOperation] = ()


@final
class _MemoryFact:
    __slots__ = (
        "identifier",
        "name",
        "summary",
    )

    identifier: str
    name: str
    summary: str

    def __init__(
        self,
        *,
        identifier: str,
        name: str,
        summary: str,
    ) -> None:
        self.identifier = identifier
        self.name = name
        self.summary = summary

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        if not isinstance(other, _MemoryFact):
            return False

        return (
            self.identifier == other.identifier
            and self.name == other.name
            and self.summary == other.summary
        )

    def __hash__(self) -> int:
        return hash((self.identifier, self.name, self.summary))

    def __repr__(self) -> str:
        return (
            f"_MemoryFact(identifier={self.identifier!r},"
            f" name={self.name!r}, summary={self.summary!r})"
        )


@final
class _MemoryRelation:
    __slots__ = (
        "label",
        "source_identifier",
        "source_name",
        "source_summary",
        "target_identifier",
        "target_name",
        "target_summary",
    )

    source_identifier: str
    source_name: str
    source_summary: str
    target_identifier: str
    target_name: str
    target_summary: str
    label: str

    def __init__(
        self,
        *,
        source_identifier: str,
        source_name: str,
        source_summary: str,
        target_identifier: str,
        target_name: str,
        target_summary: str,
        label: str,
    ) -> None:
        self.source_identifier = source_identifier
        self.source_name = source_name
        self.source_summary = source_summary
        self.target_identifier = target_identifier
        self.target_name = target_name
        self.target_summary = target_summary
        self.label = label

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        if not isinstance(other, _MemoryRelation):
            return False

        return (
            self.source_identifier == other.source_identifier
            and self.source_name == other.source_name
            and self.source_summary == other.source_summary
            and self.target_identifier == other.target_identifier
            and self.target_name == other.target_name
            and self.target_summary == other.target_summary
            and self.label == other.label
        )

    def __hash__(self) -> int:
        return hash((self.source_identifier, self.target_identifier, self.label))

    def __repr__(self) -> str:
        return (
            f"_MemoryRelation({self.source_name!r} --{self.label!r}--> {self.target_name!r},"
            f" source_identifier={self.source_identifier!r},"
            f" target_identifier={self.target_identifier!r})"
        )


@final
class _MemoryEvidence:
    __slots__ = (
        "facts",
        "relations",
    )

    facts: Sequence[_MemoryFact]
    relations: Sequence[_MemoryRelation]

    def __init__(
        self,
        *,
        facts: Sequence[_MemoryFact],
        relations: Sequence[_MemoryRelation],
    ) -> None:
        self.facts = facts
        self.relations = relations

    def __repr__(self) -> str:
        return f"_MemoryEvidence(facts={self.facts!r}, relations={self.relations!r})"


@final
class SurrealAgentMemory:
    """SurrealDB-backed, layered knowledge-graph agent memory.

    Unlike a plain conversation transcript store, this implementation maintains
    three separate knowledge graphs, one per retention layer:

    - **long_term** graph: durable knowledge owned by a single agent identity and
      shared across all of its conversation threads.
    - **mid_term** graph: knowledge scoped to a single conversation thread, shared
      between every agent participating in that thread.
    - **short_term** graph: knowledge specific to one agent within one conversation
      thread.

    ``remember`` looks up existing entities relevant to the latest exchange and asks
    an LLM to turn the exchange into explicit ``remember`` / ``forget`` operations
    against those graphs, rather than blindly appending new nodes. This
    keeps each graph deduplicated and internally consistent over long-term,
    continued usage: repeated mentions of the same entity refine one node instead of
    creating copies, and contradicted facts are forgotten instead of coexisting with
    their replacement - a same-turn ``remember`` for an entity always wins over a
    ``forget`` of that same entity, so a correction can never be swallowed by its own
    deletion. New relations are cross-referenced against both newly added and
    pre-existing entities, so the graph stays connected across ``remember`` calls
    instead of forming disconnected per-turn islands; when an entity's retention is
    reassessed and it moves to a different graph, its existing relations are carried
    over to the destination graph for every counterpart entity that still exists
    there. Nodes that are actually used - created, updated, inspected by exact name
    during recall, or referenced as a relation endpoint - have their freshness
    timestamp bumped; similarity-search recall results additionally count only once
    ``score_threshold`` turns "nearest" into a real relevance signal, and merely
    being shown to the extraction model as an existing-entity merge candidate never
    does. Each graph is capped to ``max_nodes_per_scope`` entries, pruning the least
    recently touched nodes first so long-running threads/agents do not grow memory
    without bound. The entity-extraction instructions driving ``remember`` can be
    extended per agent through ``memory_guidelines`` (a plain string or a
    ``Template``), letting different agents apply different judgment about what is
    worth remembering. Non-text parts of the exchange (e.g. images) are preserved
    and passed to the extraction model alongside the transcript, so an image-only
    exchange can still be remembered.

    ``recall`` runs an LLM-driven memory gathering loop
    (``Step.looping_completion`` over graph-aware ``find_memory`` and
    ``inspect_memory`` tools) against the incoming message: the model sees the full,
    possibly multimodal message - including image-only messages, which are not
    short-circuited away - and gathers relevant long-term, mid-term, and
    short-term continuation knowledge, including directly connected relations,
    before rewriting the model context down to a single ``ModelInput`` that
    rephrases the original message augmented with whatever it found.

    Examples
    --------
    ```python
    from draive.agents import AgentIdentity
    from draive.surreal.agent_memory import SurrealAgentMemory

    memory = SurrealAgentMemory.prepare(AgentIdentity.of(name="assistant"))
    ```
    """

    @staticmethod
    def prepare(
        identity: AgentIdentity,
        *,
        search_limit: int = 5,
        search_effort: int = 40,
        score_threshold: float | None = None,
        existing_lookup_limit: int = 8,
        max_nodes_per_scope: int | None = 500,
        memory_guidelines: Template | ModelInstructions | None = None,
        meta: Meta | MetaValues | None = None,
    ) -> AgentMemory:
        """Prepare layered knowledge-graph memory backed by SurrealDB.

        Parameters
        ----------
        identity : AgentIdentity
            Identity of the agent owning the long-term graph.
        search_limit : int, default=5
            Maximum number of seed nodes retrieved per graph on each ``search_memory``
            call during recall - the recall search loop may call it more than once.
        search_effort : int, default=40
            HNSW effort passed to SurrealDB's KNN operator during search.
        score_threshold : float | None, default=None
            Minimum cosine similarity (``1 - distance``, in ``[-1, 1]``) a node must
            reach to be treated as relevant, applied both when retrieving knowledge
            for recall and when looking up existing entities for remember. ``None``
            (the default) always uses the nearest ``search_limit``/
            ``existing_lookup_limit`` nodes in a graph regardless of how distant they
            actually are, so every non-empty graph surfaces *some* "relevant" node.
            Because that alone is not a trustworthy relevance signal, similarity-based
            recall (the ``find_memory`` tool) only counts towards
            ``max_nodes_per_scope`` recency once a real threshold is set - exact-name
            inspection (``inspect_memory``) always counts, since an exact name match
            is a genuine relevance signal on its own. Setting a threshold (for example
            ``0.2``) also lets clearly unrelated nodes be skipped from what recall
            surfaces and from what the extraction model sees as a merge candidate.
        existing_lookup_limit : int, default=8
            Maximum number of existing seed nodes per graph shown to the extraction
            model when deciding remember/forget operations during remember. Being
            shown as a merge candidate here does not by itself count as "using" a
            node - see ``max_nodes_per_scope``.
        max_nodes_per_scope : int | None, default=500
            Maximum number of nodes retained per graph per scope key. When exceeded
            after an update, the least recently touched nodes are pruned. "Touched"
            means genuinely used this turn: created, updated, inspected by exact name
            during recall, surfaced by similarity search (only once
            ``score_threshold`` is set - see above), or referenced as a relation
            endpoint - not merely shown to the extraction model as an existing-entity
            merge candidate, which would otherwise let a handful of nodes near an
            active graph's embedding dodge pruning indefinitely while newer,
            actually-referenced facts get evicted instead. ``None`` disables capacity
            enforcement.
        memory_guidelines : Template | ModelInstructions | None, default=None
            Additional, agent-specific guidance appended to the entity-extraction
            instructions used by ``remember`` - e.g. domain vocabulary, what is worth
            remembering versus ignoring for this particular agent, or retention hints.
            A ``Template`` is resolved via ``TemplatesRepository`` on every ``remember``
            call, so its content may vary at runtime; a plain string is used as-is.
            ``None`` (the default) keeps the base extraction instructions unchanged.
            Recall is not affected by this setting.
        meta : Meta | MetaValues | None, default=None
            Additional metadata attached to the resulting memory instance.

        Returns
        -------
        AgentMemory
            A configured agent memory instance backed by three SurrealDB graphs.

        Raises
        ------
        Exception
            Raised by memory operations when SurrealDB interactions fail.
        """
        agent_uri: str = identity.uri

        async def recall(
            thread: AgentThread,
            input: ModelInput,  # noqa: A002
            **extra: Any,
        ) -> ModelContext:
            _ = extra
            return await _recall_context(
                agent_uri=agent_uri,
                thread=thread,
                input=input,
                search_limit=search_limit,
                search_effort=search_effort,
                score_threshold=score_threshold,
            )

        async def remember(
            thread: AgentThread,
            context: ModelContext,
            **extra: Any,
        ) -> None:
            _ = extra
            await _remember_context(
                agent_uri=agent_uri,
                thread=thread,
                context=context,
                search_effort=search_effort,
                score_threshold=score_threshold,
                existing_lookup_limit=existing_lookup_limit,
                max_nodes_per_scope=max_nodes_per_scope,
                memory_guidelines=memory_guidelines,
            )

        return AgentMemory(
            recalling=recall,
            remembering=remember,
            meta=Meta.of(meta),
        )

    __slots__ = ()

    def __init__(self) -> NoReturn:
        raise RuntimeError("SurrealAgentMemory instantiation is forbidden")


async def _recall_context(
    *,
    agent_uri: str,
    thread: AgentThread,
    input: ModelInput,  # noqa: A002
    search_limit: int,
    search_effort: int,
    score_threshold: float | None,
) -> ModelContext:
    if not input.content:
        return (input,)

    scope_keys: Mapping[MemoryLayer, str] = _scope_keys(agent_uri, thread.identifier)

    @tool(
        name="find_memory",
        description=(
            "Find memory entities relevant to a short natural-language query across all "
            "three memory layers. Use this for fuzzy references, thread continuation, and "
            "initial discovery. Results include graph relations when available."
        ),
    )
    async def find_memory(query: str) -> MultimodalContent:
        embedded_query: Embedded[str] = await TextEmbedding.embed(query)

        # Index DDL runs sequentially before the gathered searches - concurrent
        # `DEFINE INDEX` statements destabilize the embedded engine (verified live),
        # and `_search_graph` requires the index to exist (see the comment there).
        for layer in _RECALL_LAYERS:
            await _ensure_vector_index(
                _NODE_TABLE[layer],
                dimensions=len(embedded_query.vector),
            )

        # The three graphs are independent (separate tables), so they are queried
        # concurrently instead of paying for three sequential round-trip chains.
        evidences: Sequence[_MemoryEvidence] = await asyncio.gather(
            *(
                _find_graph_evidence(
                    node_table=_NODE_TABLE[layer],
                    relation_table=_RELATION_TABLE[layer],
                    scope_key=scope_keys[layer],
                    query_vector=embedded_query.vector,
                    limit=search_limit,
                    search_effort=search_effort,
                    score_threshold=score_threshold,
                )
                for layer in _RECALL_LAYERS
            )
        )
        retrieved: dict[MemoryLayer, _MemoryEvidence] = {
            layer: evidence
            for layer, evidence in zip(_RECALL_LAYERS, evidences, strict=True)
            if evidence.facts
        }

        if not retrieved:
            return MultimodalContent.of("No relevant memory found.")

        prioritized: Mapping[MemoryLayer, _MemoryEvidence] = _prioritize_memory_evidence(retrieved)
        await _touch_memory_evidence(prioritized, score_threshold=score_threshold)
        return MultimodalContent.of(_format_memory_evidence(prioritized))

    @tool(
        name="inspect_memory",
        description=(
            "Inspect a named memory entity across all three memory layers and return its "
            "stored summary plus directly connected graph relations. Use this after "
            "finding an entity or when the incoming message names something explicitly."
        ),
    )
    async def inspect_memory(name: str) -> MultimodalContent:
        # The three graphs are independent (separate tables), so they are queried
        # concurrently instead of paying for three sequential round-trip chains.
        evidences: Sequence[_MemoryEvidence] = await asyncio.gather(
            *(
                _inspect_graph_evidence(
                    node_table=_NODE_TABLE[layer],
                    relation_table=_RELATION_TABLE[layer],
                    scope_key=scope_keys[layer],
                    name=name,
                    limit=search_limit,
                    score_threshold=score_threshold,
                )
                for layer in _RECALL_LAYERS
            )
        )
        retrieved: dict[MemoryLayer, _MemoryEvidence] = {
            layer: evidence
            for layer, evidence in zip(_RECALL_LAYERS, evidences, strict=True)
            if evidence.facts
        }

        if not retrieved:
            return MultimodalContent.of(f"No memory entity found for: {name}")

        prioritized: Mapping[MemoryLayer, _MemoryEvidence] = _prioritize_memory_evidence(retrieved)
        await _touch_memory_evidence(prioritized, score_threshold=score_threshold)
        return MultimodalContent.of(_format_memory_evidence(prioritized))

    rephrased: MultimodalContent = await Step.looping_completion(
        instructions=_RECALL_SEARCH_INSTRUCTIONS,
        tools=Toolbox.of(find_memory, inspect_memory, suggesting=find_memory),
        output="text",
    ).run((ModelInput.of(input.content),))

    if not rephrased:
        # The completion loop can legitimately end without emitting any content (e.g.
        # a provider stopping immediately with nothing to say) - never let that
        # translate into replacing the incoming request with nothing. Return the original
        # message as-is (it already carries every part, text and non-text) instead of
        # falling back to `input.content` and then re-appending its own non-text parts
        # on top of itself below, which would duplicate them.
        ctx.log_warning("Recall rewrite produced no content - using the original message as-is.")
        return (input,)

    non_text_parts = tuple(
        part for part in input.content.parts if not isinstance(part, TextContent)
    )
    return (
        ModelInput.of(
            MultimodalContent.of(rephrased, *non_text_parts),
            meta=input.meta,
        ),
    )


# In-process write locks keyed by graph scope key. A remember update is a
# read-modify-write over shared graphs: existing summaries are read, an LLM rewrites
# them, and the result is written back. Two concurrent `remember()` calls sharing a
# graph (e.g. two agents of one thread refining the same mid-term entity) would each
# rewrite from the summary they read, and the later write silently discards the
# earlier one (verified live - a logged team decision vanished this way). Serializing
# whole remember updates per scope key closes that lost-update window within one
# process; separate processes sharing one SurrealDB can still race and need
# out-of-process coordination. Weak values keep the registry from growing with every
# scope key ever seen - a lock lives exactly as long as some in-flight remember
# references it.
_GRAPH_WRITE_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def _graph_write_lock(
    scope_key: str,
    /,
) -> asyncio.Lock:
    lock: asyncio.Lock | None = _GRAPH_WRITE_LOCKS.get(scope_key)
    if lock is None:
        lock = asyncio.Lock()
        _GRAPH_WRITE_LOCKS[scope_key] = lock

    return lock


async def _remember_context(
    *,
    agent_uri: str,
    thread: AgentThread,
    context: ModelContext,
    search_effort: int,
    score_threshold: float | None,
    existing_lookup_limit: int,
    max_nodes_per_scope: int | None,
    memory_guidelines: Template | ModelInstructions | None,
) -> None:
    # Locks are acquired in sorted key order, so two remembers sharing any subset of
    # scope keys can never deadlock on each other.
    locks: list[asyncio.Lock] = [
        _graph_write_lock(key)
        for key in sorted(set(_scope_keys(agent_uri, thread.identifier).values()))
    ]
    async with AsyncExitStack() as stack:
        for lock in locks:
            await stack.enter_async_context(lock)

        await _remember_context_locked(
            agent_uri=agent_uri,
            thread=thread,
            context=context,
            search_effort=search_effort,
            score_threshold=score_threshold,
            existing_lookup_limit=existing_lookup_limit,
            max_nodes_per_scope=max_nodes_per_scope,
            memory_guidelines=memory_guidelines,
        )


async def _remember_context_locked(  # noqa: C901, PLR0912
    *,
    agent_uri: str,
    thread: AgentThread,
    context: ModelContext,
    search_effort: int,
    score_threshold: float | None,
    existing_lookup_limit: int,
    max_nodes_per_scope: int | None,
    memory_guidelines: Template | ModelInstructions | None,
) -> None:
    transcript: str
    media_parts: Sequence[MultimodalContentPart]
    transcript, media_parts = _context_transcript(context)
    if not transcript and not media_parts:
        return

    scope_keys: Mapping[MemoryLayer, str] = _scope_keys(agent_uri, thread.identifier)

    # A vector query needs text to embed - an image-only exchange has none, so the
    # similarity-gated lookup below can't run for it. Falling back to each graph's most
    # recently touched entities (a cheap, non-vector, scope_key-indexed query) still
    # gives the extraction model real merge candidates, instead of it always seeing an
    # empty <existing_memory> block for image-only turns and creating disconnected
    # duplicates of entities already remembered in a previous, text-bearing turn.
    existing: dict[MemoryLayer, Sequence[_MemoryFact]]
    node_ids: dict[tuple[MemoryLayer, str], str]
    if transcript:
        embedded_transcript: Embedded[str] = await TextEmbedding.embed(transcript)
        existing, node_ids = await _lookup_existing_entities(
            scope_keys=scope_keys,
            query_vector=embedded_transcript.vector,
            limit=existing_lookup_limit,
            search_effort=search_effort,
            score_threshold=score_threshold,
        )

    else:
        existing, node_ids = await _list_recent_entities(
            scope_keys=scope_keys,
            limit=existing_lookup_limit,
        )

    # Entities and relations are extracted in two separate passes rather than one. A
    # single combined pass asks the model to both invent entity names and consistently
    # reference those same names in relations within one response - the dominant failure
    # mode for smaller/less capable models is emitting relations that reference names it
    # never declared as entities, which then get silently dropped in `_apply_relation_
    # operations`. Extracting relations afterwards, against a closed list of names now known
    # to exist (pre-existing plus whatever entities were just remembered), removes the
    # need for the model to invent and consistently reuse a name in the same breath.
    entity_extraction: EntityExtraction = await ModelGeneration.generate(
        EntityExtraction,
        instructions=await _resolve_entity_extraction_instructions(memory_guidelines),
        input=MultimodalContent.of(_extraction_input(transcript, existing), *media_parts),
        schema_injection="simplified",
    )

    deletions, updates, additions, migrated = _classify_entity_operations(
        entity_extraction.entities, node_ids
    )

    # Capture a migrating entity's existing relations before its old node (and, with
    # it, its relation rows) gets deleted below - `_apply_entity_deletions` cascades
    # to `DELETE {relation_table} WHERE in = $_record OR out = $_record`, so anything
    # not captured first is unrecoverable.
    migrated_relations: dict[str, Sequence[_MemoryRelation]] = {}
    if migrated:
        captured: Sequence[Sequence[_MemoryRelation]] = await asyncio.gather(
            *(
                _relations_for_records(
                    relation_table=_RELATION_TABLE[old_layer],
                    seeds=[RecordID(_NODE_TABLE[old_layer], old_identifier)],
                    limit=_MAX_MIGRATED_RELATIONS_PER_ENTITY,
                )
                for old_layer, old_identifier, _ in migrated.values()
            )
        )
        migrated_relations = dict(zip(migrated.keys(), captured, strict=True))
        for name, relations in migrated_relations.items():
            if len(relations) >= _MAX_MIGRATED_RELATIONS_PER_ENTITY:
                # `_MAX_MIGRATED_RELATIONS_PER_ENTITY` is a dedicated cap, independent of
                # `existing_lookup_limit` (which bounds an unrelated thing: how many
                # existing nodes the extraction model sees as merge candidates) - surface
                # it explicitly when it's actually the limiting factor, unlike the silent
                # truncation this used to be.
                ctx.log_warning(
                    f"Migrated entity {name!r} has at least "
                    f"{_MAX_MIGRATED_RELATIONS_PER_ENTITY} relations in its old layer - "
                    "some may not have been carried over to its new layer."
                )

    await _apply_entity_deletions(deletions, node_ids=node_ids)
    await _apply_entity_updates(updates)
    touched_layers: set[MemoryLayer] = await _apply_entity_additions(
        additions,
        scope_keys=scope_keys,
        node_ids=node_ids,
    )

    # Relations only ever connect two entities within the same graph (`_apply_relation_
    # operations` enforces this), so the closed list of names shown to the model must be
    # scoped to one graph at a time - a flat, cross-layer list would let the model pick a
    # source/target pair that live in different graphs, and that relation would then be
    # silently unrepresentable. Extraction is scoped per graph, against only that graph's
    # known names, guaranteeing every name the model sees resolves within it. The graphs
    # are otherwise independent (separate relation tables), so their extraction calls run
    # concurrently instead of paying for up to three sequential model round trips.
    #
    # The closed list itself must not be limited to `node_ids` as populated above: that is
    # a *similarity-gated* subset (whatever this turn's transcript embedding happened to
    # retrieve as "existing", plus whatever was just remembered). A brand-new fact whose
    # wording doesn't happen to embed close to an already-known entity's stored text (e.g.
    # a first-person message that never names the person a new fact is about) would
    # otherwise never see that entity in its closed list, so a real, warranted relation to
    # it could never even be proposed - not because it was rejected, but because it was
    # never offered. `_RELATION_EXTRACTION_INSTRUCTIONS` already promises the model "a
    # closed list of entity names that currently exist in memory", so fetch the actual
    # membership of each graph (a cheap, non-vector, scope_key-indexed lookup) and use
    # that as the source of truth for both the prompt and endpoint resolution below.
    # `_graph_entity_names` runs after the additions above and returns each graph's
    # current membership up to `_MAX_RELATION_ENTITY_NAMES` (most recently touched
    # first), so under capacity-managed defaults it is a strict superset of the
    # similarity-gated `node_ids` for that layer.
    relation_node_ids: dict[tuple[MemoryLayer, str], str] = dict(node_ids)
    # A capacity-managed graph is already bounded by its configured cap, so use that as
    # the closed-list limit - this turn's additions may briefly exceed it, clipping only
    # the stalest entities that capacity pruning below is about to evict anyway. Only an
    # uncapped deployment needs the artificial `_MAX_RELATION_ENTITY_NAMES` bound to
    # keep the extraction prompt finite.
    names_limit: int = (
        max_nodes_per_scope if max_nodes_per_scope is not None else _MAX_RELATION_ENTITY_NAMES
    )
    # The three graphs are independent (separate tables), so this membership lookup -
    # like the KNN searches in `_lookup_existing_entities` above - runs concurrently
    # instead of paying for three sequential DB round trips.
    layer_name_rows: Sequence[Mapping[str, str]] = await asyncio.gather(
        *(
            _graph_entity_names(
                table=_NODE_TABLE[layer],
                scope_key=scope_keys[layer],
                limit=names_limit,
            )
            for layer in _LAYERS
        )
    )
    layer_name_maps: dict[MemoryLayer, dict[str, str]] = {}
    for layer, names in zip(_LAYERS, layer_name_rows, strict=True):
        if max_nodes_per_scope is None and len(names) >= _MAX_RELATION_ENTITY_NAMES:
            ctx.log_warning(
                f"The {layer!r} graph holds at least {_MAX_RELATION_ENTITY_NAMES} entities "
                "for this scope key - the relation closed list and endpoint resolution may "
                "be truncated to the most recently touched ones."
            )

        layer_names_map: dict[str, str] = dict(names)
        # `node_ids` may know about an entity this turn's `_graph_entity_names` call
        # doesn't reflect (a fake/stale backend in tests, a backend without strict
        # same-session read-after-write, or an entity truncated away by
        # `_MAX_RELATION_ENTITY_NAMES`) - keep it as a safety-net seed rather than
        # relying on the DB query alone.
        for (name_layer, name), identifier in node_ids.items():
            if name_layer == layer:
                layer_names_map.setdefault(name, identifier)

        layer_name_maps[layer] = layer_names_map
        for name, identifier in layer_names_map.items():
            relation_node_ids.setdefault((layer, name), identifier)

    eligible_layers: list[tuple[MemoryLayer, set[str]]] = [
        (layer, set(names))
        for layer, names in layer_name_maps.items()
        if len(names) >= _MIN_ENTITIES_FOR_RELATIONS
    ]

    if migrated_relations:
        await _carry_over_migrated_relations(
            migrated=migrated,
            migrated_relations=migrated_relations,
            layer_name_maps=layer_name_maps,
            relation_node_ids=relation_node_ids,
        )

    if eligible_layers:
        await asyncio.gather(
            *(
                _extract_and_apply_relations(
                    transcript=transcript,
                    media_parts=media_parts,
                    layer=layer,
                    names=names,
                    node_ids=relation_node_ids,
                )
                for layer, names in eligible_layers
            )
        )

    if max_nodes_per_scope is not None:
        for layer in touched_layers:
            await _enforce_graph_capacity(
                table=_NODE_TABLE[layer],
                relation_table=_RELATION_TABLE[layer],
                scope_key=scope_keys[layer],
                max_nodes=max_nodes_per_scope,
            )

    ctx.log_debug("...agent memory graphs updated in SurrealDB.")


async def _extract_and_apply_relations(
    *,
    transcript: str,
    media_parts: Sequence[MultimodalContentPart],
    layer: MemoryLayer,
    names: set[str],
    node_ids: Mapping[tuple[MemoryLayer, str], str],
) -> None:
    relation_extraction: RelationExtraction = await ModelGeneration.generate(
        RelationExtraction,
        instructions=_RELATION_EXTRACTION_INSTRUCTIONS,
        input=MultimodalContent.of(
            _relation_extraction_input(transcript, layer, names), *media_parts
        ),
        schema_injection="simplified",
    )
    await _apply_relation_operations(
        relation_extraction.relations,
        layer=layer,
        node_ids=node_ids,
    )


async def _carry_over_migrated_relations(
    *,
    migrated: Mapping[str, tuple[MemoryLayer, str, MemoryLayer]],
    migrated_relations: Mapping[str, Sequence[_MemoryRelation]],
    layer_name_maps: Mapping[MemoryLayer, Mapping[str, str]],
    relation_node_ids: Mapping[tuple[MemoryLayer, str], str],
) -> None:
    # A migrated entity's old relations can only be preserved for endpoints that
    # also exist, by name, in the destination graph - a different graph's relation
    # table cannot reference a node that lives in another table. Reusing
    # `_apply_relation_operations` here means the deterministic edge ids and recency
    # touch apply exactly as they would for a freshly extracted relation, so a
    # relation carried over here and a relation the same-turn extraction pass
    # independently re-derives from the transcript converge on the same edge record
    # instead of duplicating.
    for name, (_, _, new_layer) in migrated.items():
        captured: Sequence[_MemoryRelation] | None = migrated_relations.get(name)
        if not captured:
            continue

        if relation_node_ids.get((new_layer, name)) is None:
            continue  # the entity itself failed to (re)appear in its destination graph

        destination_names: Mapping[str, str] = layer_name_maps.get(new_layer, {})
        carried_over: list[RelationOperation] = []
        for relation in captured:
            is_source: bool = relation.source_name == name
            other_name: str = relation.target_name if is_source else relation.source_name
            if other_name not in destination_names:
                ctx.log_debug(
                    f"Dropping relation {relation.label!r} for migrated entity {name!r} - "
                    f"{other_name!r} does not exist in the {new_layer!r} graph."
                )
                continue

            carried_over.append(
                RelationOperation(
                    operation="add",
                    source=name if is_source else other_name,
                    target=other_name if is_source else name,
                    label=relation.label,
                )
            )

        if carried_over:
            await _apply_relation_operations(
                carried_over,
                layer=new_layer,
                node_ids=relation_node_ids,
            )


async def _lookup_existing_entities(
    *,
    scope_keys: Mapping[MemoryLayer, str],
    query_vector: Sequence[float],
    limit: int,
    search_effort: int,
    score_threshold: float | None,
) -> tuple[dict[MemoryLayer, Sequence[_MemoryFact]], dict[tuple[MemoryLayer, str], str]]:
    # Index DDL runs sequentially before the gathered searches - concurrent
    # `DEFINE INDEX` statements destabilize the embedded engine (verified live),
    # and `_search_graph` requires the index to exist (see the comment there).
    for layer in _LAYERS:
        await _ensure_vector_index(
            _NODE_TABLE[layer],
            dimensions=len(query_vector),
        )

    # The three graphs are independent (separate tables/indexes), so their KNN searches
    # run concurrently instead of paying for three sequential DB round trips.
    facts_per_layer: Sequence[Sequence[_MemoryFact]] = await asyncio.gather(
        *(
            _search_graph(
                node_table=_NODE_TABLE[layer],
                scope_key=scope_keys[layer],
                query_vector=query_vector,
                limit=limit,
                search_effort=search_effort,
                score_threshold=score_threshold,
            )
            for layer in _LAYERS
        )
    )
    return _index_layer_facts(facts_per_layer)


async def _list_recent_entities(
    *,
    scope_keys: Mapping[MemoryLayer, str],
    limit: int,
) -> tuple[dict[MemoryLayer, Sequence[_MemoryFact]], dict[tuple[MemoryLayer, str], str]]:
    # Used in place of `_lookup_existing_entities` when there is no transcript text to
    # embed for a similarity search (e.g. an image-only exchange) - falls back to each
    # graph's most recently touched entities so the extraction model still gets real
    # merge candidates instead of an empty "existing memory" view.
    facts_per_layer: Sequence[Sequence[_MemoryFact]] = await asyncio.gather(
        *(
            _list_graph_entities(
                table=_NODE_TABLE[layer],
                scope_key=scope_keys[layer],
                limit=limit,
            )
            for layer in _LAYERS
        )
    )
    return _index_layer_facts(facts_per_layer)


def _index_layer_facts(
    facts_per_layer: Sequence[Sequence[_MemoryFact]],
    /,
) -> tuple[dict[MemoryLayer, Sequence[_MemoryFact]], dict[tuple[MemoryLayer, str], str]]:
    existing: dict[MemoryLayer, Sequence[_MemoryFact]] = {}
    node_ids: dict[tuple[MemoryLayer, str], str] = {}
    for layer, facts in zip(_LAYERS, facts_per_layer, strict=True):
        if not facts:
            continue

        existing[layer] = facts
        for fact in facts:
            node_ids[(layer, fact.name)] = fact.identifier

    return existing, node_ids


async def _list_graph_entities(
    *,
    table: str,
    scope_key: str,
    limit: int,
) -> Sequence[_MemoryFact]:
    # Plain equality lookup on the indexed `scope_key` field, ordered by recency - no
    # KNN/vector search involved, used as the existing-entity merge-candidate set when
    # there is no query text to embed. `updated` must stay in the projection: SurrealDB
    # rejects `ORDER BY` when its idiom is missing from the selection.
    rows: Sequence[SurrealObject] = await Surreal.execute(
        f"""
        SELECT id, name, summary, updated FROM {table}
        WHERE scope_key = $scope_key
        ORDER BY updated DESC
        LIMIT $limit;
        """,  # nosec: B608
        scope_key=scope_key,
        limit=limit,
    )
    return tuple(
        _MemoryFact(
            identifier=_record_key(row["id"]),
            name=cast(str, row["name"]),
            summary=cast(str, row["summary"]),
        )
        for row in rows
        if isinstance(row.get("name"), str) and isinstance(row.get("summary"), str)
    )


async def _graph_entity_names(
    *,
    table: str,
    scope_key: str,
    limit: int,
) -> Mapping[str, str]:
    # Plain equality lookup on the indexed `scope_key` field - no KNN/vector search
    # involved, so this stays cheap even though it (deliberately) isn't relevance-limited.
    # Ordered by recency and capped at `limit` so an uncapped (`max_nodes_per_scope=None`)
    # deployment cannot grow the relation closed list into an arbitrarily large prompt;
    # up to the cap this is the graph's complete current membership. `updated` must stay
    # in the projection: SurrealDB rejects `ORDER BY` when its idiom is missing from the
    # selection.
    rows: Sequence[SurrealObject] = await Surreal.execute(
        f"SELECT id, name, updated FROM {table} WHERE scope_key = $scope_key "  # nosec: B608
        "ORDER BY updated DESC LIMIT $limit;",
        scope_key=scope_key,
        limit=limit,
    )
    return {cast(str, row["name"]): _record_key(row["id"]) for row in rows}


def _classify_entity_operations(  # noqa: C901
    operations: Sequence[EntityOperation],
    node_ids: Mapping[tuple[MemoryLayer, str], str],
    /,
) -> tuple[
    list[tuple[MemoryLayer, str, str]],
    list[tuple[MemoryLayer, str, EntityOperation]],
    list[EntityOperation],
    Mapping[str, tuple[MemoryLayer, str, MemoryLayer]],
]:
    forgets: list[tuple[MemoryLayer, str, str]] = []
    updates: list[tuple[MemoryLayer, str, EntityOperation]] = []
    additions: list[EntityOperation] = []
    # Entity name -> (old_layer, old_identifier, new_layer) for entities that moved
    # graphs this turn, so the caller can try to carry their existing relations over
    # to the destination graph instead of losing them outright.
    migrated: dict[str, tuple[MemoryLayer, str, MemoryLayer]] = {}

    # Every forget-list addition - whether from an explicit "forget" or from a
    # layer migration below - goes through this single dedup, keyed by the actual
    # DB record rather than by name. Funnelling layer-migration forgets through the
    # same dedup as explicit forgets matters: two "remember" operations for the same
    # name that both resolve to the same (still-unmutated) current layer but
    # different target layers previously produced two identical forget entries,
    # and the second `del node_ids[...]` in `_apply_entity_deletions` raised a
    # `KeyError`.
    forgotten: set[tuple[MemoryLayer, str]] = set()

    def _forget(layer: MemoryLayer, identifier: str, name: str, /) -> None:
        key = (layer, identifier)
        if key in forgotten:
            return

        forgotten.add(key)
        forgets.append((layer, identifier, name))

    # Only one "remember" per name can be honored in a turn: two of them may disagree
    # about retention, and letting both through would queue an in-place update against
    # the very record a same-turn migration deletes (the update then silently no-ops on
    # the deleted record and its summary is lost). Keep the latest occurrence - later
    # operations reflect the model's later judgment - and surface the conflict.
    remembers: dict[str, EntityOperation] = {}
    for operation in operations:
        if operation.operation != "remember":
            continue

        if operation.name in remembers:
            ctx.log_warning(
                f"Multiple remember operations for {operation.name!r} in one turn - "
                "keeping only the latest one."
            )

        remembers[operation.name] = operation

    for operation in remembers.values():
        target_layer: MemoryLayer = operation.retention
        # Prefer the layer named by the operation's own retention: entity
        # names are only unique within a single graph, so the same name can exist
        # in more than one graph at once (e.g. an unrelated long-term and a
        # thread-local entity happening to share a label). Without this, resolution
        # would always silently favour long-term (cross-conversation)
        # memory, letting a thread-local remember corrupt unrelated permanent
        # memory purely because of a name collision.
        resolved = _resolve_existing(
            node_ids,
            operation.name,
            preferred_layer=target_layer,
        )

        if resolved is None:
            additions.append(operation)
            continue

        current_layer, identifier = resolved
        if current_layer != target_layer:
            # Retention was re-assessed since the entity was first recorded (e.g. a
            # short-term observation turned out to be a durable, long-term fact). A
            # node's table is fixed at creation, so "moving" it between graphs means
            # deleting the old copy and re-adding it fresh in the target graph - the
            # caller uses `migrated` to try to carry its existing relations over to
            # whichever destination-graph entities they still connect to.
            _forget(current_layer, identifier, operation.name)
            migrated[operation.name] = (current_layer, identifier, target_layer)
            additions.append(operation)

        else:
            # "remember" for an already-known name becomes an update under the hood,
            # keeping the model-facing interface small without allowing duplicate nodes.
            updates.append((current_layer, identifier, operation))

    # "remember" always wins over "forget" for the same entity: every record that an
    # in-place update targets is kept, so a same-turn "forget" for that exact entity is
    # ignored below instead of deleting the correction right after writing it. Without
    # this, a model that (despite `_ENTITY_EXTRACTION_INSTRUCTIONS` telling it not to)
    # pairs a "forget" with a "remember" of the same name would have the deletion
    # silently swallow the correction, since deletions are applied before updates in
    # `_remember_context` and a plain `UPDATE` on an already-deleted record is a no-op.
    kept_from_forget: set[tuple[MemoryLayer, str]] = {
        (layer, identifier) for layer, identifier, _ in updates
    }

    for operation in operations:
        if operation.operation != "forget":
            continue

        matches = _matching_existing(node_ids, operation.name)
        if len(matches) > 1:
            # A name is only guaranteed unique within a single graph (see the
            # preferred-layer comment above) - cascading a forget across every
            # layer that currently holds it is only correct when it is genuinely
            # the same entity remembered redundantly at multiple retention layers.
            # Surface it so an unintended cross-layer deletion caused by two
            # distinct, same-named entities is at least observable.
            ctx.log_warning(
                f"Forgetting {operation.name!r} matches {len(matches)} layers at once - "
                "deleting it from all of them."
            )

        for forgotten_layer, identifier in matches:
            if (forgotten_layer, identifier) in kept_from_forget:
                continue  # a "remember" for this exact entity wins over "forget"

            _forget(forgotten_layer, identifier, operation.name)

    return forgets, updates, additions, migrated


async def _apply_entity_deletions(
    deletions: Sequence[tuple[MemoryLayer, str, str]],
    /,
    *,
    node_ids: dict[tuple[MemoryLayer, str], str],
) -> None:
    if not deletions:
        return

    # Deletions target distinct records (deduped during classification), so they are
    # independent and can run concurrently.
    await asyncio.gather(
        *(
            _delete_node(
                table=_NODE_TABLE[layer],
                relation_table=_RELATION_TABLE[layer],
                key=identifier,
            )
            for layer, identifier, _ in deletions
        )
    )
    for layer, _, name in deletions:
        del node_ids[(layer, name)]


async def _apply_entity_updates(
    updates: Sequence[tuple[MemoryLayer, str, EntityOperation]],
    /,
) -> None:
    if not updates:
        return

    embeddings: Sequence[Embedded[str]] = await TextEmbedding.embed_many(
        [f"{operation.name}: {operation.summary}" for _, _, operation in updates]
    )
    # Updates target distinct records (one "remember" per name survives classification),
    # so they are independent and can run concurrently.
    await asyncio.gather(
        *(
            _update_node(
                table=_NODE_TABLE[layer],
                key=identifier,
                summary=operation.summary,
                embedding=embedded.vector,
            )
            for (layer, identifier, operation), embedded in zip(updates, embeddings, strict=True)
        )
    )


async def _apply_entity_additions(
    additions: Sequence[EntityOperation],
    /,
    *,
    scope_keys: Mapping[MemoryLayer, str],
    node_ids: dict[tuple[MemoryLayer, str], str],
) -> set[MemoryLayer]:
    touched_layers: set[MemoryLayer] = set()
    for layer in _LAYERS:
        layer_additions = tuple(
            operation for operation in additions if operation.retention == layer
        )
        if not layer_additions:
            continue

        embeddings: Sequence[Embedded[str]] = await TextEmbedding.embed_many(
            [f"{operation.name}: {operation.summary}" for operation in layer_additions]
        )
        await _ensure_vector_index(
            _NODE_TABLE[layer],
            dimensions=len(embeddings[0].vector),
        )
        # Additions target distinct records (one "remember" per name survives
        # classification), so they are independent and can run concurrently.
        identifiers: Sequence[str] = await asyncio.gather(
            *(
                _create_node(
                    table=_NODE_TABLE[layer],
                    scope_key=scope_keys[layer],
                    name=operation.name,
                    summary=operation.summary,
                    embedding=embedded.vector,
                )
                for operation, embedded in zip(layer_additions, embeddings, strict=True)
            )
        )
        for operation, identifier in zip(layer_additions, identifiers, strict=True):
            node_ids[(layer, operation.name)] = identifier

        touched_layers.add(layer)

    return touched_layers


async def _apply_relation_operations(
    relations: Sequence[RelationOperation],
    /,
    *,
    layer: MemoryLayer,
    node_ids: Mapping[tuple[MemoryLayer, str], str],
) -> None:
    node_table: str = _NODE_TABLE[layer]
    relation_table: str = _RELATION_TABLE[layer]

    resolved: list[tuple[RelationOperation, RecordID, RecordID]] = []
    for relation in relations:
        source_identifier: str | None = node_ids.get((layer, relation.source))
        target_identifier: str | None = node_ids.get((layer, relation.target))
        if source_identifier is None or target_identifier is None:
            # Should not normally happen - the model was given a closed list of names
            # scoped to exactly this graph - but surface it instead of guessing in case
            # it invents or misremembers a name.
            ctx.log_warning(
                f"Dropping relation referencing unknown entity in {layer!r} graph: "
                f"{relation.source!r} -> {relation.target!r}"
            )
            continue

        resolved.append(
            (
                relation,
                RecordID(node_table, source_identifier),
                RecordID(node_table, target_identifier),
            )
        )

    if not resolved:
        return

    # Every endpoint the model named in a relation operation is genuine use - unlike the
    # dedup lookup in `_lookup_existing_entities` - so it counts towards
    # `max_nodes_per_scope` recency even when its own summary wasn't remembered/updated
    # this turn. Batched into one deduplicated update instead of one round trip per
    # relation.
    endpoints: dict[str, RecordID] = {}
    for _, source_record, target_record in resolved:
        endpoints.setdefault(str(source_record), source_record)
        endpoints.setdefault(str(target_record), target_record)

    await _touch_nodes(table=node_table, ids=list(endpoints.values()))

    # Mutations stay sequential rather than gathered: the model may emit a delete and an
    # add touching the same edge in one turn (a changed relationship), and their relative
    # order is meaningful - relation volume per turn is small enough not to matter.
    for relation, source_record, target_record in resolved:
        if relation.operation == "delete":
            await _delete_relations(
                relation_table=relation_table,
                source=source_record,
                target=target_record,
                label=relation.label or None,
            )
            continue

        await _create_relation(
            relation_table=relation_table,
            identifier=_relation_record_id(source_record, target_record, relation.label),
            source=source_record,
            target=target_record,
            label=relation.label,
        )


def _scope_keys(
    agent_uri: str,
    thread_identifier: UUID,
    /,
) -> Mapping[MemoryLayer, str]:
    return {
        "long_term": agent_uri,
        "mid_term": str(thread_identifier),
        "short_term": f"{agent_uri}|{thread_identifier}",
    }


def _resolve_existing(
    node_ids: Mapping[tuple[MemoryLayer, str], str],
    name: str,
    /,
    *,
    preferred_layer: MemoryLayer | None = None,
) -> tuple[MemoryLayer, str] | None:
    matches: tuple[tuple[MemoryLayer, str], ...] = _matching_existing(node_ids, name)
    if not matches:
        return None

    if preferred_layer is not None:
        for layer, identifier in matches:
            if layer == preferred_layer:
                return layer, identifier

    return matches[0]


def _matching_existing(
    node_ids: Mapping[tuple[MemoryLayer, str], str],
    name: str,
    /,
) -> tuple[tuple[MemoryLayer, str], ...]:
    return tuple(
        (layer, identifier)
        for layer in _LAYERS
        if (identifier := node_ids.get((layer, name))) is not None
    )


def _context_transcript(
    context: ModelContext,
    /,
) -> tuple[str, Sequence[MultimodalContentPart]]:
    lines: list[str] = []
    media_parts: list[MultimodalContentPart] = []
    for element in context:
        # Deliberately not "user"/"assistant": this is agent memory, and the counterpart
        # of an exchange may be a person or another agent - `AgentMessage` carries no
        # sender identity to say which. "incoming" vs "this agent" is what extraction
        # actually needs: whose assertion a fact is, and whose perspective the memory
        # belongs to.
        role: str = "incoming" if isinstance(element, ModelInput) else "this agent"
        non_text_parts: tuple[MultimodalContentPart, ...] = tuple(
            part for part in element.content.parts if not isinstance(part, TextContent)
        )
        if non_text_parts:
            # Tag each element's own attachments with their origin before handing them
            # to the extraction model - without this, an attachment generated by this
            # agent and one received from its counterpart are indistinguishable once
            # flattened into one list, and a fact drawn from the former could get
            # misattributed to the counterpart.
            media_parts.append(TextContent.of(f"[{role} attachment]"))
            media_parts.extend(non_text_parts)

        text: str = element.content.to_str().strip()
        if not text:
            continue

        lines.append(f"{role}: {text}")

    # Non-text parts (e.g. images) are collected across the whole exchange rather
    # than dropped: an image-only turn should still be usable by entity/relation
    # extraction, not silently skipped just because there is no text to embed or
    # log a transcript line for.
    return "\n".join(lines), tuple(media_parts)


def _format_knowledge(
    facts: Mapping[MemoryLayer, Sequence[_MemoryFact]],
    /,
) -> str:
    sections: list[str] = []
    for layer in _LAYERS:
        layer_facts: Sequence[_MemoryFact] | None = facts.get(layer)
        if not layer_facts:
            continue

        # Names are quoted so their boundary against the summary is unambiguous - the
        # extraction model must reuse them character-for-character (without the quotes),
        # never echo a whole line back as a new entity name.
        lines: str = "\n".join(f'- "{fact.name}": {fact.summary}' for fact in layer_facts)
        sections.append(f"{_LAYER_LABELS[layer]}:\n{lines}")

    return "\n\n".join(sections)


def _format_memory_evidence(
    evidence: Mapping[MemoryLayer, _MemoryEvidence],
    /,
) -> str:
    sections: list[str] = []
    for layer in _RECALL_LAYERS:
        layer_evidence: _MemoryEvidence | None = evidence.get(layer)
        if layer_evidence is None or not layer_evidence.facts:
            continue

        lines: list[str] = [_LAYER_LABELS[layer], "Entities:"]
        lines.extend(f"- {fact.name}: {fact.summary}" for fact in layer_evidence.facts)
        if layer_evidence.relations:
            lines.append("Relations:")
            lines.extend(
                f"- {relation.source_name} --{relation.label}--> {relation.target_name}"
                for relation in layer_evidence.relations
            )

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


async def _touch_memory_evidence(
    evidence: Mapping[MemoryLayer, _MemoryEvidence],
    /,
    *,
    score_threshold: float | None,
) -> None:
    # Without a relevance floor, "nearest" carries no guarantee that a result is
    # actually about this query - in a graph at or under `search_limit`, an unfiltered
    # KNN scan returns *something* every single call regardless of true relevance.
    # Counting that towards `max_nodes_per_scope` recency would let arbitrary nodes
    # dodge pruning just for being the least bad match this turn. Only once
    # `score_threshold` makes "found" mean "actually relevant" does similarity-derived
    # evidence (KNN seeds and their relation-expanded neighbors) count towards recency;
    # exact-name inspection seeds are the exception and are touched unconditionally in
    # `_inspect_graph_evidence`, since an exact name match is a genuine relevance
    # signal on its own. By default recency is otherwise driven purely by remember's
    # writes and relation references (see `_apply_relation_operations`, `_create_node`,
    # `_update_node`).
    if score_threshold is None:
        return

    for layer, layer_evidence in evidence.items():
        if not layer_evidence.facts:
            continue

        table: str = _NODE_TABLE[layer]
        await _touch_nodes(
            table=table,
            ids=[RecordID(table, fact.identifier) for fact in layer_evidence.facts],
        )


def _prioritize_memory_evidence(
    evidence: Mapping[MemoryLayer, _MemoryEvidence],
    /,
) -> Mapping[MemoryLayer, _MemoryEvidence]:
    # Entity names are only unique *within* a single graph (see the preferred-layer
    # comment in `_classify_entity_operations`) - two distinct, unrelated entities in
    # different layers can legitimately share a name. Deduplication therefore only
    # ever collapses exact repeats of the same DB record (matching `identifier`); it
    # never merges facts or relations across (or within) layers purely because they
    # share a name, which would risk silently dropping a distinct entity, or one of
    # its relations, just because another record happened to surface with the same
    # name first. Precedence between genuinely-the-same entity recalled at multiple
    # retention layers is left to `_RECALL_SEARCH_INSTRUCTIONS`, which already tells
    # the recall model to prefer the higher-priority layer and not blend duplicates
    # into the rewrite - the model sees every layer's evidence and decides, instead
    # of the code deciding for it by deleting data it hasn't seen yet.
    prioritized: dict[MemoryLayer, _MemoryEvidence] = {}

    for layer in _RECALL_LAYERS:
        layer_evidence: _MemoryEvidence | None = evidence.get(layer)
        if layer_evidence is None:
            continue

        seen_facts: set[str] = set()
        layer_facts: list[_MemoryFact] = []
        for fact in layer_evidence.facts:
            if fact.identifier in seen_facts:
                continue

            seen_facts.add(fact.identifier)
            layer_facts.append(fact)

        known_identifiers: set[str] = {fact.identifier for fact in layer_facts}
        seen_relations: set[tuple[str, str, str]] = set()
        layer_relations: list[_MemoryRelation] = []
        for relation in layer_evidence.relations:
            if (
                relation.source_identifier not in known_identifiers
                or relation.target_identifier not in known_identifiers
            ):
                continue

            relation_key: tuple[str, str, str] = _memory_relation_key(relation)
            if relation_key in seen_relations:
                continue

            seen_relations.add(relation_key)
            layer_relations.append(relation)

        if layer_facts:
            prioritized[layer] = _MemoryEvidence(
                facts=tuple(layer_facts),
                relations=tuple(layer_relations),
            )

    return prioritized


def _normalized_label(
    label: str,
    /,
) -> str:
    # The single normalization used everywhere a relation label is compared or hashed -
    # existence checks, deduplication, deterministic edge ids - so case/whitespace
    # variants of one label ("Works at" vs "works at ") always denote the same edge.
    return label.strip().casefold()


def _memory_relation_key(
    relation: _MemoryRelation,
    /,
) -> tuple[str, str, str]:
    return (
        relation.source_identifier,
        _normalized_label(relation.label),
        relation.target_identifier,
    )


async def _resolve_entity_extraction_instructions(
    memory_guidelines: Template | ModelInstructions | None,
    /,
) -> ModelInstructions:
    if memory_guidelines is None:
        return _ENTITY_EXTRACTION_INSTRUCTIONS

    resolved_guidelines: str
    if isinstance(memory_guidelines, Template):
        # `default=""` avoids an uncaught `TemplateMissing` when no `TemplatesRepository`
        # able to resolve this identifier is configured in the active context - falling
        # through to the empty-string check below degrades to the base instructions
        # instead of crashing every single `remember()` call.
        resolved_guidelines = await TemplatesRepository.resolve_str(
            memory_guidelines,
            default="",
        )

    else:
        resolved_guidelines = memory_guidelines

    if not resolved_guidelines.strip():
        return _ENTITY_EXTRACTION_INSTRUCTIONS

    return (
        f"{_ENTITY_EXTRACTION_INSTRUCTIONS}\n\n"
        "<agent_guidelines>\n"
        "Additional memory guidelines for this agent. Apply them especially to long-term "
        "agent-specific knowledge; they take precedence if they conflict with the rules "
        "above:\n"
        f"{resolved_guidelines}\n"
        "</agent_guidelines>"
    )


def _extraction_input(
    transcript: str,
    existing: Mapping[MemoryLayer, Sequence[_MemoryFact]],
    /,
) -> str:
    # Delimited exactly as `_ENTITY_EXTRACTION_INSTRUCTIONS` describes its input: the
    # quoted-name line format inside <existing_memory> is what lets the model tell an
    # entity's exact name apart from its summary (echoing a whole "name: summary" line
    # back as a new entity name was a verified live failure mode).
    existing_block: str = _format_knowledge(existing) if existing else "none yet"
    return (
        f"<exchange>\n{transcript}\n</exchange>\n\n"
        f"<existing_memory>\n{existing_block}\n</existing_memory>"
    )


def _relation_extraction_input(
    transcript: str,
    layer: MemoryLayer,
    names: Iterable[str],
    /,
) -> str:
    # One name per line: names may themselves contain commas, so a comma-joined list
    # would make the closed list's element boundaries ambiguous.
    names_block: str = "\n".join(sorted(names))
    return (
        f"<exchange>\n{transcript}\n</exchange>\n\n"
        f"Memory layer: {_LAYER_LABELS[layer]}.\n"
        f"<entities>\n{names_block}\n</entities>"
    )


def _record_key(
    value: SurrealValue,
    /,
) -> str:
    if isinstance(value, SurrealID):
        return str(value.record)

    if isinstance(value, RecordID):
        return str(value.id)

    if isinstance(value, str):
        return value

    raise SurrealException(f"Unexpected SurrealDB record identifier: {value!r}")


def _retryable_conflict(
    exception: Exception,
    /,
) -> bool:
    # SurrealDB marks transient optimistic-transaction conflicts as explicitly
    # retryable ("read or write conflict ... can be retried") - verified live with
    # gathered statements racing each other.
    return isinstance(exception, SurrealException) and "can be retried" in str(exception)


@retry(limit=2, delay=lambda attempt, _: 0.05 * attempt, catching=_retryable_conflict)
async def _ensure_vector_index(
    table: str,
    /,
    *,
    dimensions: int,
) -> None:
    # Without a HNSW index defined on `embedding`, SurrealDB's `<|K,EF|>` operator
    # doesn't error - it silently returns nothing (verified live on the embedded
    # engine), indistinguishable from a genuine no-match result. Every caller that runs
    # a KNN search (not just node creation) must therefore ensure the index exists
    # first. `DEFINE INDEX IF NOT EXISTS` is idempotent and cheap, and is deliberately
    # NOT cached in-process: `Surreal` is context state, so a single process can talk
    # to several namespaces/databases, and a cache keyed by table name alone would skip
    # index creation on every database after the first - silently losing recall there.
    await Surreal.execute(
        f"DEFINE INDEX IF NOT EXISTS {table}_embedding_index "
        f"ON TABLE {table} FIELDS embedding "
        f"HNSW DIMENSION {dimensions} TYPE F64 DIST COSINE;"
    )


# Every mutation helper below carries the same conflict retry as `_ensure_vector_index`:
# they are gathered concurrently against shared tables, and on server-backed engines
# racing statements surface as retryable optimistic-transaction conflicts (verified
# live - previously only the index DDL was guarded and a single conflict crashed the
# whole `remember()` update). All of them are idempotent (deterministic record ids,
# UPSERT/UPDATE/DELETE-by-key semantics), so a retry after a half-applied attempt
# converges on the same state.
@retry(limit=2, delay=lambda attempt, _: 0.05 * attempt, catching=_retryable_conflict)
async def _touch_nodes(
    *,
    table: str,
    ids: Sequence[RecordID],
) -> None:
    if not ids:
        return

    await Surreal.execute(
        f"UPDATE {table} SET updated = time::now() WHERE id IN $ids;",  # nosec: B608
        ids=list(ids),
    )


def _entity_record_id(
    scope_key: str,
    name: str,
    /,
) -> str:
    # Deterministic, derived from the (scope_key, name) logical key instead of random: this
    # lets `_create_node` UPSERT instead of CREATE, so concurrent `remember()` calls racing to
    # add the same new entity collapse onto one row instead of creating duplicate nodes.
    return _record_digest(scope_key, name)


@retry(limit=2, delay=lambda attempt, _: 0.05 * attempt, catching=_retryable_conflict)
async def _create_node(
    *,
    table: str,
    scope_key: str,
    name: str,
    summary: str,
    embedding: Sequence[float],
) -> str:
    identifier: str = _entity_record_id(scope_key, name)
    await Surreal.execute(
        """
        UPSERT $_record CONTENT {
            scope_key: $scope_key,
            name: $name,
            summary: $summary,
            embedding: $embedding,
            created: time::now(),
            updated: time::now()
        };
        """,
        _record=RecordID(table, identifier),
        scope_key=scope_key,
        name=name,
        summary=summary,
        embedding=list(embedding),
    )

    return identifier


@retry(limit=2, delay=lambda attempt, _: 0.05 * attempt, catching=_retryable_conflict)
async def _update_node(
    *,
    table: str,
    key: str,
    summary: str,
    embedding: Sequence[float],
) -> None:
    await Surreal.execute(
        "UPDATE $_record SET summary = $summary, embedding = $embedding, updated = time::now();",
        _record=RecordID(table, key),
        summary=summary,
        embedding=list(embedding),
    )


@retry(limit=2, delay=lambda attempt, _: 0.05 * attempt, catching=_retryable_conflict)
async def _delete_node(
    *,
    table: str,
    relation_table: str,
    key: str,
) -> None:
    record = RecordID(table, key)
    await Surreal.execute(
        f"DELETE {relation_table} WHERE in = $_record OR out = $_record;",  # nosec: B608
        _record=record,
    )
    await Surreal.execute(
        f"DELETE {table} WHERE id = $_record;",  # nosec: B608
        _record=record,
    )


def _record_digest(*parts: str) -> str:
    # Shared digest for deterministic record identifiers derived from logical keys.
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _relation_record_id(
    source: RecordID,
    target: RecordID,
    label: str,
    /,
) -> str:
    # Deterministic, derived from (source, target, normalized label) just like
    # `_entity_record_id` is for nodes: duplicate adds of the same relation - whether
    # from concurrent `remember()` calls or repeated extraction across turns - collapse
    # onto one edge record instead of inserting duplicate parallel edges.
    return _record_digest(str(source), str(target), _normalized_label(label))


@retry(limit=2, delay=lambda attempt, _: 0.05 * attempt, catching=_retryable_conflict)
async def _create_relation(
    *,
    relation_table: str,
    identifier: str,
    source: RecordID,
    target: RecordID,
    label: str,
) -> None:
    # RELATE with an explicit, deterministic edge id (see `_relation_record_id`) - the
    # id is a hex digest generated locally, so interpolating it is safe. Verified against
    # embedded SurrealDB: RELATE onto an existing edge id replaces that edge record in
    # place, making adds idempotent without any existence pre-check - duplicate adds and
    # concurrent `remember()` calls converge on a single edge instead of inserting
    # duplicates.
    try:
        await Surreal.execute(
            f"RELATE $_source->{relation_table}:⟨{identifier}⟩->$_target "  # nosec: B608
            "CONTENT { label: $_label, created: time::now() };",
            _source=source,
            _target=target,
            _label=label,
        )

    except SurrealException as exc:
        if "already exist" not in str(exc).casefold():
            raise

        # Guard for server versions that treat RELATE onto an existing id as a
        # CREATE-style conflict instead of a replace - either way the edge this call
        # wanted is already there, so losing the race is not worth failing the whole
        # memory update over.
        ctx.log_debug(
            f"Relation {relation_table}:{identifier} already exists - skipping duplicate."
        )


@retry(limit=2, delay=lambda attempt, _: 0.05 * attempt, catching=_retryable_conflict)
async def _delete_relations(
    *,
    relation_table: str,
    source: RecordID,
    target: RecordID,
    label: str | None,
) -> None:
    if label:
        # Labels are compared normalized (matching `_normalized_label`, which also feeds
        # the deterministic edge ids), so a case/whitespace variant of a stored label is
        # still deletable by that label.
        await Surreal.execute(
            f"DELETE {relation_table} "  # nosec: B608
            "WHERE in = $_source AND out = $_target "
            "AND string::lowercase(string::trim(label)) = $_label;",
            _source=source,
            _target=target,
            _label=_normalized_label(label),
        )

    else:
        await Surreal.execute(
            f"DELETE {relation_table} WHERE in = $_source AND out = $_target;",  # nosec: B608
            _source=source,
            _target=target,
        )


async def _enforce_graph_capacity(
    *,
    table: str,
    relation_table: str,
    scope_key: str,
    max_nodes: int,
) -> None:
    # `START` skips the `max_nodes` freshest rows, so only the prunable overflow tail is
    # ever fetched - not the graph's entire membership on every enforcement pass. The
    # `updated` field must stay in the projection: SurrealDB rejects `START` when the
    # `ORDER BY` idiom is missing from the selection.
    rows: Sequence[SurrealObject] = await Surreal.execute(
        f"""
        SELECT id, updated FROM {table}
        WHERE scope_key = $scope_key
        ORDER BY updated DESC
        START $start;
        """,  # nosec: B608
        scope_key=scope_key,
        start=max_nodes,
    )
    if not rows:
        return

    await asyncio.gather(
        *(
            _delete_node(
                table=table,
                relation_table=relation_table,
                key=_record_key(row["id"]),
            )
            for row in rows
        )
    )


def _filter_by_score(
    rows: Sequence[SurrealObject],
    score_threshold: float | None,
    /,
) -> Sequence[SurrealObject]:
    if score_threshold is None:
        return rows

    filtered: list[SurrealObject] = []
    for row in rows:
        distance: SurrealValue = row.get("distance")
        if not isinstance(distance, int | float):
            continue

        if (1.0 - float(distance)) >= score_threshold:
            filtered.append(row)

    return filtered


async def _find_graph_evidence(
    *,
    node_table: str,
    relation_table: str,
    scope_key: str,
    query_vector: Sequence[float],
    limit: int,
    search_effort: int,
    score_threshold: float | None,
) -> _MemoryEvidence:
    seed_facts: Sequence[_MemoryFact] = await _search_graph(
        node_table=node_table,
        scope_key=scope_key,
        query_vector=query_vector,
        limit=limit,
        search_effort=search_effort,
        score_threshold=score_threshold,
    )
    return await _graph_evidence_for_facts(
        node_table=node_table,
        relation_table=relation_table,
        facts=seed_facts,
        limit=limit,
    )


async def _inspect_graph_evidence(
    *,
    node_table: str,
    relation_table: str,
    scope_key: str,
    name: str,
    limit: int,
    score_threshold: float | None,
) -> _MemoryEvidence:
    rows: Sequence[SurrealObject] = await Surreal.execute(
        f"""
        SELECT
            id,
            name,
            summary

        FROM
            {node_table}

        WHERE
            scope_key = $scope_key
        AND
            name = $name

        LIMIT $limit;
        """,  # nosec: B608
        scope_key=scope_key,
        name=name,
        limit=limit,
    )

    seed_facts: Sequence[_MemoryFact] = tuple(
        _MemoryFact(
            identifier=_record_key(row["id"]),
            name=cast(str, row["name"]),
            summary=cast(str, row["summary"]),
        )
        for row in rows
        if isinstance(row.get("name"), str) and isinstance(row.get("summary"), str)
    )
    if seed_facts and score_threshold is None:
        # Unlike similarity-derived recall evidence (gated by `score_threshold` in
        # `_touch_memory_evidence`), an exact-name inspection is a genuine relevance
        # signal by itself, so its seeds count towards `max_nodes_per_scope` recency
        # even without a threshold. With a threshold set, `_touch_memory_evidence`
        # already touches the full evidence (seeds included) - skipping the local touch
        # avoids writing the same rows twice per inspect call.
        await _touch_nodes(
            table=node_table,
            ids=[RecordID(node_table, fact.identifier) for fact in seed_facts],
        )

    return await _graph_evidence_for_facts(
        node_table=node_table,
        relation_table=relation_table,
        facts=seed_facts,
        limit=limit,
    )


async def _graph_evidence_for_facts(
    *,
    node_table: str,
    relation_table: str,
    facts: Sequence[_MemoryFact],
    limit: int,
) -> _MemoryEvidence:
    if not facts:
        return _MemoryEvidence(facts=(), relations=())

    relations: Sequence[_MemoryRelation] = await _relations_for_records(
        relation_table=relation_table,
        seeds=[RecordID(node_table, fact.identifier) for fact in facts],
        limit=limit,
    )

    related_facts: list[_MemoryFact] = list(facts)
    known_identifiers: set[str] = {fact.identifier for fact in related_facts}
    for relation in relations:
        if relation.source_identifier not in known_identifiers:
            related_facts.append(
                _MemoryFact(
                    identifier=relation.source_identifier,
                    name=relation.source_name,
                    summary=relation.source_summary,
                )
            )
            known_identifiers.add(relation.source_identifier)

        if relation.target_identifier not in known_identifiers:
            related_facts.append(
                _MemoryFact(
                    identifier=relation.target_identifier,
                    name=relation.target_name,
                    summary=relation.target_summary,
                )
            )
            known_identifiers.add(relation.target_identifier)

    return _MemoryEvidence(
        facts=tuple(related_facts),
        relations=relations,
    )


async def _relations_for_records(
    *,
    relation_table: str,
    seeds: Sequence[RecordID],
    limit: int,
) -> Sequence[_MemoryRelation]:
    if not seeds:
        return ()

    # Fetched per seed (concurrently) rather than with one shared `LIMIT` across all
    # seeds combined: a single global limit lets a highly-connected hub entity (e.g.
    # the central person or project of a thread) dominate the result set and crowd
    # out the other seeds' relations entirely. Capping and fetching per seed guarantees
    # every seed gets its own budget, and running the per-seed queries concurrently
    # keeps the wall-clock cost the same as the old single-query version.
    rows_per_seed: Sequence[Sequence[SurrealObject]] = await asyncio.gather(
        *(
            _relations_for_seed(relation_table=relation_table, seed=seed, limit=limit)
            for seed in seeds
        )
    )

    seen: set[tuple[str, str, str]] = set()
    deduplicated: list[_MemoryRelation] = []
    # Merged round-robin across seeds (one row from each seed per round) instead of
    # concatenating every seed's full `limit` rows in sequence - each seed still gets a
    # fair shot at contributing, but the merged, deduplicated result stays bounded by
    # `limit` overall instead of growing up to `limit * len(seeds)`.
    max_rows_per_seed: int = max((len(rows) for rows in rows_per_seed), default=0)
    for index in range(max_rows_per_seed):
        for rows in rows_per_seed:
            if index >= len(rows):
                continue

            relation: _MemoryRelation | None = _memory_relation_from_row(rows[index])
            if relation is None:
                continue

            key = _memory_relation_key(relation)
            if key in seen:
                continue

            seen.add(key)
            deduplicated.append(relation)
            if len(deduplicated) >= limit:
                return tuple(deduplicated)

    return tuple(deduplicated)


async def _relations_for_seed(
    *,
    relation_table: str,
    seed: RecordID,
    limit: int,
) -> Sequence[SurrealObject]:
    return await Surreal.execute(
        f"""
        SELECT
            in.{{id, name, summary}} AS source,
            out.{{id, name, summary}} AS target,
            label

        FROM
            {relation_table}

        WHERE
            in = $_seed
        OR
            out = $_seed

        LIMIT $limit;
        """,  # nosec: B608
        _seed=seed,
        limit=limit,
    )


def _memory_relation_from_row(
    row: SurrealObject,
    /,
) -> _MemoryRelation | None:
    source: SurrealValue = row.get("source")
    target: SurrealValue = row.get("target")
    label: SurrealValue = row.get("label")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        return None

    source_identifier: SurrealValue = source.get("id")
    source_name: SurrealValue = source.get("name")
    source_summary: SurrealValue = source.get("summary")
    target_identifier: SurrealValue = target.get("id")
    target_name: SurrealValue = target.get("name")
    target_summary: SurrealValue = target.get("summary")
    if (
        source_identifier is None
        or target_identifier is None
        or not isinstance(source_name, str)
        or not isinstance(source_summary, str)
        or not isinstance(target_name, str)
        or not isinstance(target_summary, str)
    ):
        return None

    return _MemoryRelation(
        source_identifier=_record_key(source_identifier),
        source_name=source_name,
        source_summary=source_summary,
        target_identifier=_record_key(target_identifier),
        target_name=target_name,
        target_summary=target_summary,
        label=label if isinstance(label, str) and label else "related to",
    )


async def _search_graph(
    *,
    node_table: str,
    scope_key: str,
    query_vector: Sequence[float],
    limit: int,
    search_effort: int,
    score_threshold: float | None,
) -> Sequence[_MemoryFact]:
    # Callers must run `_ensure_vector_index` for this table before searching: a missing
    # HNSW index doesn't error - the KNN operator silently returns nothing (verified
    # live on the embedded engine), indistinguishable from a genuine no-match result.
    # The DDL cannot live here because independent layers are searched gathered, and
    # concurrent `DEFINE INDEX` statements destabilize the embedded engine (also
    # verified live) - callers issue it sequentially before gathering.

    # The `scope_key` equality below is not a post-filter over globally-nearest rows:
    # SurrealDB pushes WHERE conditions combined with the HNSW KNN operator into the
    # index search itself (visible as a `predicate` on the `KnnScan` operator under
    # EXPLAIN), rejecting non-matching candidates during graph traversal before they
    # occupy one of the K slots - so the nearest `limit` results are the nearest within
    # this graph, even though the table is shared by every agent/thread's scope key.
    raw_rows: Sequence[SurrealObject] = await Surreal.execute(
        f"""
        SELECT
            id,
            name,
            summary,
            vector::distance::knn() AS distance

        FROM
            {node_table}

        WHERE
            scope_key = $scope_key
        AND
            embedding <|{limit},{search_effort}|> $query

        ORDER BY
            distance ASC;
        """,  # nosec: B608
        scope_key=scope_key,
        query=list(query_vector),
    )

    # The KNN operator above always returns the nearest `limit` rows in the graph, however
    # distant they actually are - filter out ones that don't clear `score_threshold` (cosine
    # similarity, since the index is always defined with COSINE distance) before they get
    # touched/expanded below, so unrelated nodes are neither surfaced nor kept artificially
    # fresh by capacity pruning.
    seed_rows: Sequence[SurrealObject] = _filter_by_score(raw_rows, score_threshold)

    if not seed_rows:
        return ()

    # `row["id"]` comes back as draive's `SurrealID` wrapper, which the driver cannot
    # re-encode as a query parameter - rebuild native `RecordID`s to reuse them below.
    # Note: this raw KNN scan is never itself "use" of a node - callers decide whether
    # and how to touch what they get back (see `_lookup_existing_entities`, which uses
    # this purely as a merge-candidate scan, and `_touch_memory_evidence`, which bumps
    # recency for similarity-derived recall evidence only once `score_threshold` is
    # set; exact-name inspection touches its seeds itself in `_inspect_graph_evidence`).
    facts: list[_MemoryFact] = [
        _MemoryFact(
            identifier=_record_key(row["id"]),
            name=cast(str, row["name"]),
            summary=cast(str, row["summary"]),
        )
        for row in seed_rows
    ]

    seen: set[str] = set()
    deduplicated: list[_MemoryFact] = []
    for fact in facts:
        if fact.identifier in seen:
            continue

        seen.add(fact.identifier)
        deduplicated.append(fact)

    return tuple(deduplicated)
