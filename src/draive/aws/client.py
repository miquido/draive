from collections.abc import Collection, Iterable
from types import TracebackType
from typing import final

from haiway import State

from draive.aws.api import AWSAPI
from draive.aws.s3 import AWSS3Mixin
from draive.resources import ResourcesRepository

__all__ = ("AWS",)


@final
class AWS(
    AWSS3Mixin,
    AWSAPI,
):
    """AWS service facade bundling S3 and repository integrations.

    Parameters
    ----------
    region_name
        Preferred AWS region. Falls back to profile or environment
        configuration when omitted.
    access_key_id
        Optional access key identifier used to override ambient
        credentials.
    secret_access_key
        Secret key paired with ``access_key_id`` when overriding
        credentials.
    features
        Collection of repository feature classes to activate while the
        client is bound in a context manager.
    """

    __slots__ = ("_features",)

    def __init__(
        self,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        features: Collection[type[ResourcesRepository]] | None = None,
    ) -> None:
        super().__init__(
            region_name=region_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )

        self._features: Collection[type[ResourcesRepository]]
        if features is not None:
            self._features = features

        else:
            self._features = (ResourcesRepository,)

    async def __aenter__(self) -> Iterable[State]:
        """Prepare the AWS client and bind selected features to context."""
        await self._prepare_client()

        if ResourcesRepository in self._features:
            return (
                ResourcesRepository(
                    fetching=self.fetch,
                    uploading=self.upload,
                ),
            )

        return ()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """No-op cleanup to satisfy the async context manager protocol."""
