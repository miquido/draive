from typing import Self, overload

from draive.parameters import DataModel

__all__ = [
    "ModelTokenUsage",
    "TokenUsage",
]


class ModelTokenUsage(DataModel):
    input_tokens: int
    output_tokens: int

    def __add__(
        self,
        other: Self,
    ) -> Self:
        return self.__class__(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class TokenUsage(DataModel):
    @overload
    @classmethod
    def for_model(
        cls,
        name: str,
        *,
        input_tokens: int | None,
    ) -> Self: ...

    @overload
    @classmethod
    def for_model(
        cls,
        name: str,
        *,
        output_tokens: int | None,
    ) -> Self: ...

    @overload
    @classmethod
    def for_model(
        cls,
        name: str,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> Self: ...

    @classmethod
    def for_model(
        cls,
        name: str,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> Self:
        return cls(
            usage={
                name: ModelTokenUsage(
                    input_tokens=input_tokens or 0,
                    output_tokens=output_tokens or 0,
                ),
            },
        )

    usage: dict[str, ModelTokenUsage]

    def __add__(
        self,
        other: Self,
    ) -> Self:
        usage: dict[str, ModelTokenUsage] = self.usage
        for key, value in other.usage.items():
            if current := usage.get(key):
                usage[key] = current + value
            else:
                usage[key] = value

        return self.__class__(usage=usage)
