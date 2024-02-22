import json

from draive.helpers import extract_parameters_specification
from draive.types.state import State

__all__ = [
    "Generated",
]


class Generated(State):
    @classmethod
    def specification(cls) -> str:
        if not hasattr(cls, "_specification"):
            cls._specification: str = json.dumps(
                extract_parameters_specification(cls.__init__),
                indent=2,
            )

        return cls._specification
