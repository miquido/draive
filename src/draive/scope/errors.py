__all__ = [
    "MissingScopeContext",
    "MissingScopeDependency",
    "MissingScopeState",
]


from typing_extensions import deprecated


class MissingScopeContext(Exception):
    pass


@deprecated("MissingScopeDependency will be removed")
class MissingScopeDependency(Exception):
    pass


class MissingScopeState(Exception):
    pass
