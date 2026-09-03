import pytest

from draive.aws.cloudwatch import format_metric_dimensions, format_metric_unit


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        # units recorded within draive
        ("count", "Count"),
        ("tokens", "None"),  # there is no unit suitable for tokens
        ("%", "Percent"),
        # units already matching CloudWatch
        ("Count", "Count"),
        ("Milliseconds", "Milliseconds"),
        ("Count/Second", "Count/Second"),
        # nothing else can be sent - it fails the whole request
        ("widgets", None),
        ("", None),
        (None, None),
    ],
)
def test_format_metric_unit_maps_only_standard_units(
    unit: str | None,
    expected: str | None,
) -> None:
    assert format_metric_unit(unit) == expected


def test_format_metric_dimensions_skips_empty_values() -> None:
    dimensions = format_metric_dimensions(
        {
            "service": "draive",
            "empty": "",
            "": "unnamed",
            "blank": (),
        }
    )

    assert dimensions == [{"Name": "service", "Value": "draive"}]


def test_format_metric_dimensions_respects_limits() -> None:
    dimensions = format_metric_dimensions(
        {f"attribute_{index}": f"value_{index}" for index in range(64)}
    )

    assert len(dimensions) == 30


def test_format_metric_dimensions_truncates_long_values() -> None:
    dimensions = format_metric_dimensions(
        {
            "a" * 512: "b" * 2048,
        }
    )

    assert len(dimensions[0]["Name"]) == 255
    assert len(dimensions[0]["Value"]) == 1024
