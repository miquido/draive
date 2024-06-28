from typing import final

from draive.parameters.data import ParametrizedData

__all__ = [
    "State",
    "Stateless",
]


State = ParametrizedData


@final
class Stateless(ParametrizedData):
    pass
