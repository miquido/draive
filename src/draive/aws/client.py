from collections.abc import Collection, Iterable
from types import TracebackType
from typing import final

from haiway import State

from draive.aws.api import AWSAPI
from draive.aws.s3 import AWSS3Mixin
from draive.aws.sqs import AWSSQSMixin
from draive.aws.state import AWSSQS
from draive.resources import ResourcesRepository

__all__ = ("AWS",)


@final
class AWS(
    AWSS3Mixin,
    AWSSQSMixin,
    AWSAPI,
):
    """AWS service facade bundling S3 and SQS integrations.

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
        Collection of feature state classes (for example
        :class:`ResourcesRepository`, :class:`AWSSQS`) to activate while the
        client is bound in a context manager.
    """

    __slots__ = ("_features",)

    def __init__(
        self,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        features: Collection[type[ResourcesRepository | AWSSQS]] | None = None,
    ) -> None:
        super().__init__(
            region_name=region_name,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
        )

        self._features: Collection[type[ResourcesRepository | AWSSQS]]
        if features is not None:
            self._features = features

        else:
            self._features = ()

    async def __aenter__(self) -> Iterable[State]:
        features: list[State] = []

        if ResourcesRepository in self._features:
            await self._prepare_s3_client()
            features.append(
                ResourcesRepository(
                    fetching=self.fetch,
                    uploading=self.upload,
                ),
            )

        if AWSSQS in self._features:
            await self._prepare_sqs_client()

            features.append(
                AWSSQS(queue_accessing=self._queue_access),
            )

        return features

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass
