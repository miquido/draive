from haiway import State

from draive.utils.attributes import attribute_path_components, attribute_path_segments


class _Leaf(State):
    value: str


class _Root(State):
    leaf: _Leaf


def test_plain_path_resolves_field_names() -> None:
    path = _Root._.leaf.value

    assert attribute_path_components(path) == ("leaf", "value")
    assert attribute_path_segments(path) == ("leaf", "value")
