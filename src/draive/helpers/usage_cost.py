from collections.abc import Mapping
from typing import Self

from haiway import State, ctx

from draive.metrics.tokens import ModelTokenUsage, TokenUsage

__all__ = [
    "ModelTokenPrice",
    "TokenPrice",
    "usage_cost",
]


class ModelTokenPrice(State):
    input_token: float
    output_token: float


class TokenPrice(State):
    @classmethod
    def of(
        cls,
        models: Mapping[str, ModelTokenPrice],
        /,
    ) -> Self:
        return cls(price=models)

    price: Mapping[str, ModelTokenPrice]

    def __add__(
        self,
        other: Self,
    ) -> Self:
        return self.__class__(
            price={  # always use newer
                **self.price,
                **other.price,
            }
        )

    def cost_for(
        self,
        model: str,
        /,
        *,
        usage: ModelTokenUsage,
    ) -> float:
        if price := self.price.get(model):
            return price.input_token * usage.input_tokens + price.output_token * usage.output_tokens

        else:
            raise ValueError(f"Missing token price for model {model}")


async def usage_cost() -> float:
    model_prices: TokenPrice = ctx.state(TokenPrice)
    model_usage: TokenUsage = await ctx.read(
        TokenUsage,
        merged=True,
        default=TokenUsage(usage={}),
    )
    cost: float = 0

    for model, usage in model_usage.usage.items():
        cost += model_prices.cost_for(model, usage=usage)

    return cost
