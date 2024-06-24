from draive.parameters.data import ParametrizedData

__all__ = [
    "State",
    "Stateless",
]


class State(ParametrizedData):
    def __str__(self) -> str:
        return str(self.as_dict(aliased=False))


class Stateless(ParametrizedData):
    pass
