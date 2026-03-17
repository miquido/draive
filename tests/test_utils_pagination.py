from draive import Paginated, Pagination


def test_paginated_sequence_behavior() -> None:
    pagination = Pagination.of(
        token="cursor-2",
        limit=32,
    )
    page = Paginated[int].of(
        [1, 2, 3],
        pagination=pagination,
    )

    assert len(page) == 3
    assert list(page) == [1, 2, 3]
    assert page[0] == 1
    assert page[1:] == (2, 3)
    assert page.has_next_page is True
    assert page.pagination == pagination
    assert page.token == "cursor-2"


def test_paginated_preserves_items_and_pagination_metadata() -> None:
    pagination = Pagination.of(
        token="cursor-3",
        limit=32,
    )
    page = Paginated[str].of(
        ["a", "b"],
        pagination=pagination,
    )

    assert page.items == ("a", "b")
    assert page.pagination == pagination
    assert page.token == "cursor-3"


def test_pagination_defaults() -> None:
    pagination = Pagination.of(limit=32)

    assert pagination.limit == 32
    assert pagination.token is None
    assert pagination.arguments == {}
