from collections.abc import Mapping
from typing import Literal, overload

from haiway import State, ctx

from draive.configuration.types import ConfigurationLoading
from draive.parameters import BasicValue, DataModel

__all__ = ("Configuration",)


async def _no_config(
    identifier: str,
) -> Mapping[str, BasicValue] | None:
    return None


class Configuration(State):
    @overload
    @classmethod
    async def load[Config: DataModel | State](
        cls,
        config: type[Config],
        /,
        *,
        key: str | None = None,
    ) -> Config | None: ...

    @overload
    @classmethod
    async def load[Config: DataModel | State](
        cls,
        config: type[Config],
        /,
        *,
        key: str | None = None,
        default: Config,
    ) -> Config: ...

    @overload
    @classmethod
    async def load[Config: DataModel | State](
        cls,
        config: type[Config],
        /,
        *,
        key: str | None = None,
        required: Literal[True],
    ) -> Config: ...

    @classmethod
    async def load[Config: DataModel | State](
        cls,
        config: type[Config],
        /,
        *,
        key: str | None = None,
        default: Config | Mapping[str, BasicValue] | None = None,
        required: bool = False,
    ) -> Config | Mapping[str, BasicValue] | None:
        identifier: str = config.__name__ if key is None else key
        loaded: Mapping[str, BasicValue] | None = await ctx.state(cls).loading(identifier)

        if loaded is not None:
            return config.from_mapping(loaded)

        elif default is not None:
            return default

        elif required:
            raise ValueError(f"Missing configuration for {identifier}")

        else:
            return None

    loading: ConfigurationLoading = _no_config
