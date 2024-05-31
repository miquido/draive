__all__ = [
    "RateLimitError",
]


class RateLimitError(Exception):
    def __init__(
        self,
        *args: object,
        retry_after: float,
    ) -> None:
        super().__init__(*args)
        self.retry_after: float = retry_after
