from typing import Any

from haiway import asynchronous

__all__ = ("BedrockAPI",)


class BedrockAPI:
    __slots__ = ("_client",)

    def __init__(self) -> None:
        self._client: Any

    # preparing it lazily on demand, boto does a lot of stuff on initialization
    @asynchronous
    def _initialize_client(self) -> None:
        if hasattr(self, "_client"):
            return  # already initialized

        # postponing import of boto3 as late as possible, it does a lot of stuff
        import boto3  # pyright: ignore[reportMissingTypeStubs]

        self._client = boto3.Session().client("bedrock-runtime")  # pyright: ignore[reportUnknownMemberType]

    def _deinitialize_client(self) -> None:
        if not hasattr(self, "_client"):
            return  # already deinitialized

        del self._client
