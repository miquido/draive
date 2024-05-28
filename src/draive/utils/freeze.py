from typing import Any

__all__ = [
    "freeze",
]


def freeze(instance: object, /) -> None:
    def frozen_set(
        __name: str,
        __value: Any,
    ) -> None:
        raise RuntimeError(f"{instance.__class__.__qualname__} is frozen and can't be modified")

    def frozen_del(
        __name: str,
    ) -> None:
        raise RuntimeError(f"{instance.__class__.__qualname__} is frozen and can't be modified")

    instance.__delattr__ = frozen_del
    instance.__setattr__ = frozen_set
