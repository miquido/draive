from typing import TypeVar

from draive.types import ParametrizedModel, ParametrizedState

__all__ = [
    "Metric",
    "Metric_T",
    "SerializableMetric",
]

Metric = ParametrizedState
SerializableMetric = ParametrizedModel


Metric_T = TypeVar(
    "Metric_T",
    bound=Metric,
)
