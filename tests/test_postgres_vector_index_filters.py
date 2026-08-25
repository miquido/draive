from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

import pytest
from haiway import Alias, AttributeRequirement, State
from haiway.postgres import PostgresValue

from draive.postgres.vector_index import postgres_identifier, resolve_requirements


class _AliasedNested(State):
    inner_value: Annotated[str, Alias("inner")]


class _Aliased(State):
    identifier: Annotated[str, Alias("id")]
    nested: _AliasedNested


class _Document(State):
    name: str
    count: int
    ratio: float
    active: bool
    identifier: UUID
    created: datetime
    tags: Sequence[str]


def test_equal_uses_parameterized_json_accessor() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Document].equal("draive", _Document._.name)
    )

    assert where_clause == "payload #> $1::TEXT[] = $2::JSONB"
    assert list(arguments) == [("name",), '"draive"']


def test_not_equal_uses_parameterized_json_accessor() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Document].not_equal("draive", _Document._.name)
    )

    assert where_clause == "(payload #> $1::TEXT[] IS DISTINCT FROM $2::JSONB)"
    assert list(arguments) == [("name",), '"draive"']


def test_non_string_values_are_compared_as_json() -> None:
    """Regression test: the `#>>` text accessor forced every compared argument to
    be text, which asyncpg rejects for any int, float or bool value.
    """
    requirements: Sequence[tuple[AttributeRequirement[_Document], str]] = (
        (AttributeRequirement[_Document].equal(3, _Document._.count), "3"),
        (AttributeRequirement[_Document].equal(0.5, _Document._.ratio), "0.5"),
        (AttributeRequirement[_Document].equal(True, _Document._.active), "true"),
        (
            AttributeRequirement[_Document].equal(UUID(int=1), _Document._.identifier),
            '"00000000-0000-0000-0000-000000000001"',
        ),
        (
            AttributeRequirement[_Document].equal(
                datetime(2026, 1, 1, tzinfo=UTC),
                _Document._.created,
            ),
            '"2026-01-01T00:00:00+00:00"',
        ),
    )
    for requirement, expected in requirements:
        where_clause: str
        arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
        where_clause, arguments = resolve_requirements(requirement)

        assert where_clause == "payload #> $1::TEXT[] = $2::JSONB"
        assert list(arguments)[1] == expected


def test_contained_in_reads_swapped_operands() -> None:
    # haiway builds `contained_in` with the collection as `lhs`
    # and the attribute path as `rhs` - unlike every other operator
    requirement: AttributeRequirement[_Document] = AttributeRequirement[_Document].contained_in(
        ["draive", "haiway"],
        _Document._.name,
    )
    assert requirement.lhs == ["draive", "haiway"]

    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(requirement)

    assert where_clause == "payload #> $1::TEXT[] = ANY($2::JSONB[])"
    assert list(arguments) == [("name",), ['"draive"', '"haiway"']]


def test_contains_any_uses_json_elements() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Document].contains_any(
            ["draive", "haiway"],
            _Document._.tags,
        )
    )

    assert where_clause == (
        "EXISTS (SELECT 1 FROM jsonb_array_elements(payload #> $1::TEXT[]) AS element"
        " WHERE element = ANY($2::JSONB[]))"
    )
    assert list(arguments) == [("tags",), ['"draive"', '"haiway"']]


def test_contains_uses_json_elements() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Document].contains(
            "draive",
            _Document._.tags,
        )
    )

    assert where_clause == (
        "EXISTS (SELECT 1 FROM jsonb_array_elements(payload #> $1::TEXT[]) AS element"
        " WHERE element = $2::JSONB)"
    )
    assert list(arguments) == [("tags",), '"draive"']


def test_collection_operators_emit_balanced_parentheses() -> None:
    requirements: Sequence[AttributeRequirement[_Document]] = (
        AttributeRequirement[_Document].contains("draive", _Document._.tags),
        AttributeRequirement[_Document].contains_any(["draive"], _Document._.tags),
        AttributeRequirement[_Document].contained_in(["draive"], _Document._.name),
    )
    for requirement in requirements:
        where_clause, _ = resolve_requirements(requirement)
        assert where_clause.count("(") == where_clause.count(")")


def test_nested_requirements_number_arguments_sequentially() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Document].contained_in(["draive"], _Document._.name)
        & AttributeRequirement[_Document].contains("haiway", _Document._.tags)
    )

    assert where_clause == (
        "(payload #> $1::TEXT[] = ANY($2::JSONB[])"
        " AND EXISTS (SELECT 1 FROM jsonb_array_elements(payload #> $3::TEXT[]) AS element"
        " WHERE element = $4::JSONB))"
    )
    assert list(arguments) == [("name",), ['"draive"'], ("tags",), '"haiway"']


def test_missing_requirements_produce_empty_clause() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(None)

    assert where_clause == ""
    assert list(arguments) == []


def test_nested_attribute_paths_become_path_segments() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Document]("meta.origin.name", "equal", "draive", check=lambda _: None)
    )

    assert where_clause == "payload #> $1::TEXT[] = $2::JSONB"
    assert list(arguments) == [("meta", "origin", "name"), '"draive"']


def test_unsafe_attribute_paths_are_rejected() -> None:
    """Regression test: the attribute path used to be interpolated into a quoted
    SQL literal without any escaping, allowing a crafted path to break out of it.
    """
    unsafe_paths: Sequence[str] = (
        "name' OR '1'='1",
        "name'}' OR TRUE --",
        'name"',
        "na me",
        "",
        "name;DROP TABLE _Document",
    )
    for path in unsafe_paths:
        with pytest.raises(ValueError, match="Invalid Postgres attribute path"):
            resolve_requirements(
                AttributeRequirement[_Document](path, "equal", "draive", check=lambda _: None)
            )


def test_unsafe_identifiers_are_rejected() -> None:
    """Regression test: table names have no parameter form and are interpolated
    into the statement, therefore each one has to be verified upfront.
    """
    assert postgres_identifier("_Document") == "_Document"
    for identifier in ("bad name", "table;DROP TABLE other", 'quoted"', "1table", ""):
        with pytest.raises(ValueError, match="Invalid Postgres identifier"):
            postgres_identifier(identifier)


def test_aliased_attributes_use_serialized_payload_keys() -> None:
    """Regression test: payloads are stored with aliases applied, filtering by the
    python attribute name silently matched nothing.
    """
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Aliased].equal("doc-1", _Aliased._.identifier)
    )

    assert where_clause == "payload #> $1::TEXT[] = $2::JSONB"
    assert list(arguments) == [("id",), '"doc-1"']


def test_nested_aliased_attributes_resolve_every_segment() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Aliased].equal("value", _Aliased._.nested.inner_value)
    )

    assert where_clause == "payload #> $1::TEXT[] = $2::JSONB"
    assert list(arguments) == [("nested", "inner"), '"value"']


def test_contained_in_resolves_aliased_attribute() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Aliased].contained_in(("doc-1", "doc-2"), _Aliased._.identifier)
    )

    assert where_clause == "payload #> $1::TEXT[] = ANY($2::JSONB[])"
    assert list(arguments) == [("id",), ['"doc-1"', '"doc-2"']]


def test_text_match_requires_every_token() -> None:
    where_clause: str
    arguments: Sequence[Sequence[PostgresValue] | PostgresValue]
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Document].text_match("Typed State", _Document._.name)
    )

    assert where_clause == (
        "(payload #>> $1::TEXT[] ~* ('\\y' || $2::TEXT || '\\y')"
        " AND payload #>> $1::TEXT[] ~* ('\\y' || $3::TEXT || '\\y'))"
    )
    assert list(arguments) == [("name",), "typed", "state"]


def test_text_match_without_tokens_matches_everything() -> None:
    where_clause: str
    where_clause, arguments = resolve_requirements(
        AttributeRequirement[_Document].text_match("...", _Document._.name)
    )

    assert where_clause == "TRUE"
    assert not list(arguments)
