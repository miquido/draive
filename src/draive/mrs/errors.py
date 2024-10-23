from typing_extensions import deprecated

__all__ = [
    "MRSException",
]


@deprecated("mistralrs support will be removed")
class MRSException(Exception):
    pass
