from collections.abc import Iterable, Mapping, MutableMapping

from haiway import State

from draive.configuration import Config, Configuration
from draive.parameters import DataModel
from draive.parameters.types import BasicValue

__all__ = ("VolatileConfiguration",)


def VolatileConfiguration(
    configuration: Iterable[
        tuple[str, DataModel]
        | tuple[str, State]
        | tuple[str, Mapping[str, BasicValue]]
        | DataModel
        | State
    ],
    /,
) -> Configuration:
    configurations: MutableMapping[str, Mapping[str, BasicValue]] = {}
    for config in configuration:
        match config:
            case Config():  # it is state although it fails to match for State
                configurations[type(config).__name__] = config.to_mapping()

            case State():
                configurations[type(config).__name__] = config.to_mapping()

            case DataModel():
                configurations[type(config).__name__] = config.to_mapping()

            case (str() as key, State() as state):
                configurations[key] = state.to_mapping()

            case (str() as key, DataModel() as model):
                configurations[key] = model.to_mapping()

            case (str() as key, mapping):
                assert isinstance(mapping, Mapping)  # nosec: B101

                configurations[key] = mapping

    async def loading(
        identifier: str,
    ) -> Mapping[str, BasicValue] | None:
        return configurations.get(identifier)

    return Configuration(loading=loading)
