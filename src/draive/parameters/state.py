from draive.parameters import ParametrizedData

__all__ = [
    "State",
]


class State(ParametrizedData):
    def __str__(self) -> str:
        return str(self.as_dict(aliased=False))
