__all__ = [
    "MissingScopeContext",
    "MissingScopeDependency",
    "MissingScopeState",
]


class MissingScopeContext(Exception):
    pass


class MissingScopeDependency(Exception):
    pass


class MissingScopeState(Exception):
    pass
