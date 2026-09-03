from collections.abc import Mapping, Sequence
from datetime import timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import pytest
from haiway import AttributeRequirement, State
from surrealdb import Duration, RecordID, Table
from surrealdb.cbor import CBORTag
from surrealdb.data.types.constants import TAG_DURATION_COMPACT

from draive.surreal import SurrealClient
from draive.surreal.filters import prepare_filter
from draive.surreal.types import SurrealID, SurrealObject, SurrealValue
from draive.surreal.utils import surreal_identifier, surreal_value, surreal_variable


class _SurrealValuesItem(State):
    id: SurrealID
    a: int
    name: str = ""


def test_surreal_value_converts_record_ids_of_all_supported_kinds() -> None:
    """Regression test: integer record identifiers used to fall through to the
    terminal error, only UUID and string ones were converted.
    """
    identifier: UUID = uuid4()

    assert surreal_value(RecordID("n", 1)) == SurrealID(table="n", record=1)
    assert surreal_value(RecordID("n", "abc")) == SurrealID(table="n", record="abc")
    assert surreal_value(RecordID("n", identifier)) == SurrealID(table="n", record=identifier)


def test_surreal_variable_converts_surreal_ids_back_to_record_ids() -> None:
    """Regression test: `SurrealID` is what we produce for record links on the way
    out, feeding it back in used to reach the terminal error instead.
    """
    converted: Any = surreal_variable(SurrealID(table="n", record="abc"))

    assert isinstance(converted, RecordID)
    assert converted.table_name == "n"
    assert converted.id == "abc"


def test_surreal_variable_converts_nested_values() -> None:
    converted: Any = surreal_variable(
        {
            "link": SurrealID(table="n", record="abc"),
            "list": (Decimal("1.5"), "text"),
        }
    )

    assert isinstance(converted["link"], RecordID)
    assert converted["list"] == [1.5, "text"]


def test_prepare_filter_uses_contained_in_operands_in_the_order_haiway_builds_them() -> None:
    """Regression test: 'contained_in' is the only operator haiway builds with its
    operands swapped, the collection is the lhs and the attribute path is the rhs.
    """
    assert prepare_filter(AttributeRequirement.contained_in((3, 5), _SurrealValuesItem._.a)) == (
        "a INSIDE $_f0",
        {"_f0": [3, 5]},
    )


def test_prepare_filter_rejects_contained_in_without_a_sequence() -> None:
    with pytest.raises(ValueError):
        prepare_filter(AttributeRequirement.contained_in("35", _SurrealValuesItem._.a))


def test_prepare_filter_casts_text_match_operands() -> None:
    """Regression test: `string(...)` is not a valid SurrealQL function path,
    values have to be cast instead.
    """
    assert prepare_filter(AttributeRequirement.text_match("alpha", _SurrealValuesItem._.name)) == (
        "string::contains(<string>name, <string>$_f0)",
        {"_f0": "alpha"},
    )


def test_surreal_value_converts_durations_to_timedelta() -> None:
    """Regression test: a decoded `Duration` used to reach the terminal error,
    making every record holding a duration field unreadable.
    """
    assert surreal_value(Duration(1_500_000_000)) == timedelta(seconds=1.5)
    assert surreal_value(Duration(0)) == timedelta()


def test_surreal_variable_encodes_durations_within_the_compact_tag() -> None:
    """Regression test: the SDK encodes its `Duration` as a [seconds, nanoseconds]
    pair under the *string* duration tag, which SurrealDB refuses to parse - the
    embedded engine fails to decode the request while a server drops it without a
    response, wedging the connection. Durations have to use the compact tag.
    """
    for value in (timedelta(seconds=1.5), Duration(1_500_000_000)):
        converted: Any = surreal_variable(value)

        assert isinstance(converted, CBORTag)
        assert converted.tag == TAG_DURATION_COMPACT
        assert converted.value == [1, 500_000_000]

    assert surreal_variable(timedelta()).value == [0, 0]


def test_surreal_value_keeps_timedelta_durations() -> None:
    assert surreal_value(timedelta(seconds=1.5)) == timedelta(seconds=1.5)


def test_surreal_value_keeps_table_names() -> None:
    """Regression test: `Table` is the only representation SurrealDB accepts for a
    table name bound as a variable, it used to reach the terminal error.
    """
    assert surreal_value(Table("n")) == Table("n")


def test_surreal_identifier_rejects_unsafe_names() -> None:
    assert surreal_identifier("_ExampleModel") == "_ExampleModel"
    for identifier in ("bad name", "table; DELETE other", 'quoted"', "1table", ""):
        with pytest.raises(ValueError, match="Invalid SurrealDB identifier"):
            surreal_identifier(identifier)


@pytest.mark.asyncio
async def test_surreal_embedded_statements_preserve_record_ids_and_scalar_results() -> None:
    """Live regression test against the embedded engine covering integer record
    identifiers, scalar statement results, the `SurrealID` round trip through a
    filter and the 'contained_in' operand order.
    """
    async with SurrealClient(
        url="mem://",
        namespace="test_surreal_values",
        database="embedded_values",
    ) as client:
        async with client.prepare_session() as session:
            identifier: UUID = uuid4()
            assert await session.execute("CREATE n:1 SET a=3;") == (
                {"a": 3, "id": SurrealID(table="n", record=1)},
            )
            assert await session.execute("CREATE n:abc SET a=3;") == (
                {"a": 3, "id": SurrealID(table="n", record="abc")},
            )
            assert await session.execute(f"CREATE n:u'{identifier}' SET a=5;") == (
                {"a": 5, "id": SurrealID(table="n", record=identifier)},
            )

            # scalar statement results used to be dropped by the summary match arm
            assert await session.execute("RETURN 42;") == ({"value": 42},)
            assert await session.execute("RETURN $v;", v=7) == ({"value": 7},)
            assert await session.execute("RETURN 'text';") == ({"value": "text"},)

            # list, record and empty results keep working as before
            assert await session.execute("SELECT VALUE a FROM n ORDER BY a;") == (
                {"value": 3},
                {"value": 3},
                {"value": 5},
            )
            assert await session.execute("SELECT count() FROM n GROUP ALL;") == ({"count": 3},)
            assert await session.execute("RETURN [];") == ()
            assert await session.execute("DEFINE TABLE IF NOT EXISTS q SCHEMALESS;") == ()

            # a record identifier taken out of the store can be filtered on again
            rows: Sequence[SurrealObject] = await session.execute("SELECT * FROM n:abc;")
            clause: str
            variables: Mapping[str, SurrealValue]
            clause, variables = prepare_filter(
                AttributeRequirement.equal(rows[0]["id"], _SurrealValuesItem._.id)
            )
            assert await session.execute(
                f"SELECT VALUE id FROM n WHERE {clause};",  # nosec: B608
                **variables,
            ) == ({"value": SurrealID(table="n", record="abc")},)

            clause, variables = prepare_filter(
                AttributeRequirement.contained_in((5,), _SurrealValuesItem._.a)
            )
            assert await session.execute(
                f"SELECT VALUE a FROM n WHERE {clause};",  # nosec: B608
                **variables,
            ) == ({"value": 5},)


@pytest.mark.asyncio
async def test_surreal_embedded_statements_round_trip_durations() -> None:
    """Live regression test: durations produced by a statement have to be usable as
    variables again. The SDK encodes its `Duration` under the string duration tag,
    which the engine refuses to parse - the embedded one fails to decode the request
    while a server drops it without any response, wedging the connection.
    """

    class _SurrealDurationItem(State):
        label: str
        span: timedelta

    async with SurrealClient(
        url="mem://",
        namespace="test_surreal_values",
        database="embedded_durations",
    ) as client:
        async with client.prepare_session() as session:
            assert await session.execute("RETURN $span;", span=timedelta(seconds=90)) == (
                {"value": timedelta(seconds=90)},
            )
            assert await session.execute("RETURN $span;", span=timedelta(microseconds=250)) == (
                {"value": timedelta(microseconds=250)},
            )
            assert await session.execute("RETURN $span;", span=Duration(1_500_000_000)) == (
                {"value": timedelta(seconds=1.5)},
            )
            # durations nested within content are converted as well
            assert await session.execute(
                "RETURN $nested;",
                nested={"spans": [timedelta(minutes=1)]},
            ) == ({"spans": [timedelta(minutes=1)]},)

            await session.create(
                _SurrealDurationItem(label="timed", span=timedelta(minutes=2)),
                identifier="timed",
            )
            page = await session.fetch(
                _SurrealDurationItem,
                requirements=AttributeRequirement.equal(
                    timedelta(minutes=2),
                    _SurrealDurationItem._.span,
                ),
            )
            assert page.items == (_SurrealDurationItem(label="timed", span=timedelta(minutes=2)),)
