from typing import Any

from haiway import asynchronous, getenv_str

__all__ = ("BedrockAPI",)


class BedrockAPI:
    __slots__ = (
        "_access_key",
        "_access_key_id",
        "_client",
        "_region_name",
    )

    def __init__(
        self,
        *,
        region_name: str | None = None,
        access_key_id: str | None = None,
        access_key: str | None = None,
    ) -> None:
        self._region_name: str | None = region_name or getenv_str("AWS_DEFAULT_REGION")
        self._access_key_id: str | None = access_key_id or getenv_str("AWS_ACCESS_KEY_ID")
        self._access_key: str | None = access_key or getenv_str("AWS_ACCESS_KEY")
        self._client: Any

    # preparing it lazily on demand, boto does a lot of stuff on initialization
    @asynchronous
    def _initialize_client(self) -> None:
        if hasattr(self, "_client"):
            return  # already initialized

        # postponing import of boto3 as late as possible, it does a lot of stuff
        import boto3  # pyright: ignore[reportMissingTypeStubs]

        self._client = boto3.Session(  # pyright: ignore[reportUnknownMemberType]
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._access_key,
            region_name=self._region_name,
        ).client("bedrock-runtime")

    def _deinitialize_client(self) -> None:
        if not hasattr(self, "_client"):
            return  # already deinitialized

        del self._client
