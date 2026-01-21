from typing import Any

from boto3 import Session  # pyright: ignore[reportMissingTypeStubs]
from haiway import asynchronous

__all__ = ("AWSAPI",)


class AWSAPI:
    """Low-level AWS session and client management.

    Provides asynchronous client initializers for AWS services so higher-level
    mixins can share a single boto3 session without duplicating configuration.
    """

    __slots__ = (
        "_cloudwatch_client",
        "_cloudwatch_logs_client",
        "_eventbridge_client",
        "_s3_client",
        "_session",
        "_sqs_client",
    )

    def __init__(
        self,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        """Create an AWS session.

        Parameters
        ----------
        region_name
            Preferred AWS region for the session. Defaults to the
            region configured in the environment or AWS profiles.
        access_key_id
            Optional access key identifier used for credential override.
        secret_access_key
            Optional secret key paired with ``access_key_id`` for override.
        profile_name
            Optional profile name to be used instead of Default profile.
        """
        # using dict as kwargs since existence of some
        # arguments when initializing session changes
        # the session configuration, even if using None
        kwargs: dict[str, object] = {}

        if key_id := access_key_id:
            kwargs["aws_access_key_id"] = key_id

        if key := secret_access_key:
            kwargs["aws_secret_access_key"] = key

        if region := region_name:
            kwargs["region_name"] = region

        if profile := profile_name:
            kwargs["profile_name"] = profile

        self._session: Session = Session(**kwargs)
        self._cloudwatch_client: Any
        self._cloudwatch_logs_client: Any
        self._eventbridge_client: Any
        self._s3_client: Any
        self._sqs_client: Any

    @asynchronous
    def _prepare_s3_client(self) -> None:
        self._s3_client = self._session.client(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            service_name="s3",
        )

    @asynchronous
    def _prepare_sqs_client(self) -> None:
        self._sqs_client = self._session.client(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            service_name="sqs",
        )

    @asynchronous
    def _prepare_cloudwatch_clients(self) -> None:
        self._cloudwatch_client = self._session.client(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            service_name="cloudwatch",
        )
        self._cloudwatch_logs_client = self._session.client(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            service_name="logs",
        )
        self._eventbridge_client = self._session.client(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            service_name="events",
        )

    @property
    def region(self) -> str | None:
        """Currently configured AWS region for the session."""

        return self._session.region_name  # pyright: ignore[reportUnknownMemberType, reportReturnType, reportUnknownVariableType]
