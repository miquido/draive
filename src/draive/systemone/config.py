from haiway import Configuration

__all__ = ("SystemOneConfig",)


class SystemOneConfig(Configuration):
    """Configuration of System One API decisions.

    Attributes
    ----------
    model : str
        Model name or alias, OpenJev servers ignore it and answer with the model they serve.
    """

    model: str = "jev-latest"
