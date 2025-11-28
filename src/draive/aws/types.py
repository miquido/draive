"""
AWS specific exception types.
"""

from __future__ import annotations

__all__ = (
    "AWSAccessDenied",
    "AWSError",
    "AWSResourceNotFound",
)


class AWSError(Exception):
    """Base error raised when an AWS request fails.

    Parameters
    ----------
    uri
        URI of the AWS resource involved in the failing request.
    code
        AWS error code when available.
    message
        Human readable error message returned by AWS.
    """
    __slots__ = (
        "code",
        "message",
        "uri",
    )

    def __init__(
        self,
        *,
        uri: str,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        detail: str = message or "AWS request failed"
        if code:
            detail = f"{detail} ({code})"

        super().__init__(f"{detail} - {uri}")
        self.uri: str = uri
        self.code: str | None = code
        self.message: str | None = message


class AWSAccessDenied(AWSError):
    """Error raised when AWS denies access to the requested resource."""

    def __init__(
        self,
        *,
        uri: str,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            code=code or "AccessDenied",
            message=message or "Access denied",
        )


class AWSResourceNotFound(AWSError):
    """Error raised when the requested AWS resource cannot be found."""

    def __init__(
        self,
        *,
        uri: str,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        super().__init__(
            uri=uri,
            code=code or "ResourceNotFound",
            message=message or "Resource not found",
        )
