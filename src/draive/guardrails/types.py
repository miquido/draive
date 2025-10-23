from haiway import Meta, MetaValues

__all__ = (
    "GuardrailsException",
    "GuardrailsFailure",
)


class GuardrailsException(Exception):
    __slots__ = ("meta",)

    def __init__(
        self,
        *args: object,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(*args)
        self.meta: Meta = Meta.of(meta)


class GuardrailsFailure(GuardrailsException):
    __slots__ = ("cause",)

    def __init__(
        self,
        *args: object,
        cause: Exception,
        meta: Meta | MetaValues | None = None,
    ) -> None:
        super().__init__(*args, meta=meta)
        self.cause: Exception = cause
