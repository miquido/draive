from draive.parameters.basic import BasicValue

__all__ = [
    "JSON",
]

JSON = dict[str, BasicValue] | list[BasicValue]
