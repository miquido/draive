from collections.abc import Mapping, Sequence
from typing import Annotated, Any

import pytest
from haiway import Alias, AttributePath, AttributeRequirement, State, Validator
from qdrant_client.models import FieldCondition

from draive.postgres.vector_index import resolve_requirements
from draive.qdrant.filters import prepare_filter as qdrant_filter
from draive.surreal.filters import prepare_filter as surreal_filter
from draive.utils.attributes import attribute_path_components, attribute_path_segments


class _Nested(State):
    value: Annotated[str, Alias("stored_value")]


class _Document(State):
    nested: Annotated[_Nested, Alias("stored_nested")]
    optional: _Nested | None
    entries: Annotated[Sequence[_Nested], Alias("stored_entries")]
    mapping: Mapping[str, _Nested]
    pair: tuple[str, _Nested]
    content: _Nested
    validated_entries: Annotated[Sequence[_Nested], Validator(lambda value: value)]


@pytest.mark.parametrize(
    ("path", "components", "key"),
    (
        (_Document._.nested.value, ("stored_nested", "stored_value"), "stored_nested.stored_value"),
        (_Document._.optional.value, ("optional", "stored_value"), "optional.stored_value"),
        (
            _Document._.entries[0].value,
            ("stored_entries", 0, "stored_value"),
            "stored_entries[0].stored_value",
        ),
        (
            _Document._.mapping["key"].value,
            ("mapping", "key", "stored_value"),
            "mapping.key.stored_value",
        ),
        (_Document._.pair[1].value, ("pair", 1, "stored_value"), "pair[1].stored_value"),
        (_Document._.content.value, ("content", "stored_value"), "content.stored_value"),
        (
            _Document._.validated_entries[2].value,
            ("validated_entries", 2, "stored_value"),
            "validated_entries[2].stored_value",
        ),
    ),
    ids=("nested", "optional", "sequence", "mapping", "tuple", "content", "validated_sequence"),
)
def test_database_filters_resolve_typed_paths(
    path: AttributePath[_Document, str],
    components: Sequence[str | int],
    key: str,
) -> None:
    assert attribute_path_components(path) == components
    segments = tuple(str(component) for component in components)
    assert attribute_path_segments(path) == segments
    requirement = AttributeRequirement[_Document].equal("match", path)
    _, arguments = resolve_requirements(requirement)
    assert arguments == [segments, '"match"']
    prepared = qdrant_filter(requirement)
    assert prepared is not None
    assert prepared.must is not None
    condition = next(iter(prepared.must))
    assert isinstance(condition, FieldCondition)
    assert condition.key == key
    assert surreal_filter(requirement) == (f"{key} = $_f0", {"_f0": "match"})
    assert surreal_filter(requirement, scoped=True) == (f"content.{key} = $_f0", {"_f0": "match"})


@pytest.mark.parametrize("path", ("nested.value", "entries[0].value", "", None))
def test_untyped_paths_are_rejected(path: Any) -> None:
    with pytest.raises(AssertionError, match="Expected an AttributePath"):
        attribute_path_segments(path)


def test_mapping_keys_are_not_split() -> None:
    assert attribute_path_segments(_Document._.mapping["a.b[c]"].value) == (
        "mapping",
        "a.b[c]",
        "stored_value",
    )
