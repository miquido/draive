__all__ = [
    "AgentException",
]


class AgentException(Exception):
    def __init__(self, cause: BaseException, *args: object) -> None:
        self.__cause__ = cause
        super().__init__(*args)
