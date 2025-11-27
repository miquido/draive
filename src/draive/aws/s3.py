import re
from asyncio import gather
from collections.abc import Collection, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urlparse

from botocore.exceptions import ClientError  # pyright: ignore[reportMissingModuleSource]
from haiway import META_EMPTY, BasicValue, Meta, asynchronous

from draive.aws.api import AWSAPI
from draive.aws.types import AWSAccessDenied, AWSError, AWSResourceNotFound
from draive.resources import ResourceContent, ResourceReference

__all__ = ("AWSS3Mixin",)


class AWSS3Mixin(AWSAPI):
    """S3 helper mixin that implements fetch, download, and upload APIs."""

    async def fetch(
        self,
        uri: str,
        **extra: Any,
    ) -> Collection[ResourceReference] | ResourceContent | None:
        """Fetch an object from S3 and wrap it in ``ResourceContent``.

        Parameters
        ----------
        uri
            S3 URI pointing to the object (``s3://bucket/key``).

        Returns
        -------
        Collection[ResourceReference] | ResourceContent | None
            Downloaded content wrapped with metadata for S3 objects.

        Raises
        ------
        ValueError
            If the ``uri`` does not use the ``s3://`` scheme.
        AWSAccessDenied
            If S3 rejects the request due to missing or invalid credentials.
        AWSResourceNotFound
            If the bucket or object key does not exist.
        AWSError
            For other S3 client failures.
        """
        if not uri.startswith("s3://"):
            raise ValueError("Unsupported fetch uri scheme")

        parsed_uri: ParseResult = urlparse(uri)  # s3://bucket/name
        bucket: str = parsed_uri.netloc
        name: str = parsed_uri.path.lstrip("/")
        content, object_info = await gather(
            self._fetch(
                bucket=bucket,
                name=name,
            ),
            self._get_object_info(
                bucket=bucket,
                name=name,
            ),
            return_exceptions=False,
        )
        mime_type, meta = object_info
        return ResourceContent.of(
            content,
            mime_type=mime_type or "application/octet-stream",
            meta=meta,
        )

    @asynchronous
    def _fetch(
        self,
        bucket: str,
        name: str,
    ) -> bytes:
        output: BytesIO = BytesIO()
        try:
            self._s3_client.download_fileobj(
                Bucket=bucket,
                Key=name,
                Fileobj=output,
            )

        except ClientError as exc:
            raise _translate_client_error(
                error=exc,
                bucket=bucket,
                key=name,
            ) from exc

        return output.getvalue()

    async def download(
        self,
        uri: str,
        /,
        *,
        save_path: Path | str,
    ) -> None:
        """Download an S3 object directly to disk.

        Parameters
        ----------
        uri
            S3 URI pointing to the object (``s3://bucket/key``).
        save_path
            Filesystem path where the object should be written.

        Raises
        ------
        ValueError
            If the ``uri`` does not use the ``s3://`` scheme.
        AWSAccessDenied
            If S3 rejects the request due to missing or invalid credentials.
        AWSResourceNotFound
            If the bucket or object key does not exist.
        AWSError
            For other S3 client failures.
        """
        if not uri.startswith("s3://"):
            raise ValueError("Unsupported download uri scheme")

        parsed_uri: ParseResult = urlparse(uri)  # s3://bucket/name
        bucket: str = parsed_uri.netloc
        name: str = parsed_uri.path.lstrip("/")
        return await self._download(
            bucket=bucket,
            name=name,
            save_path=save_path,
        )

    @asynchronous
    def _download(
        self,
        bucket: str,
        name: str,
        save_path: Path | str,
    ) -> None:
        with open(save_path, "wb") as file:
            try:
                self._s3_client.download_fileobj(
                    Bucket=bucket,
                    Key=name,
                    Fileobj=file,
                )

            except ClientError as exc:
                raise _translate_client_error(
                    error=exc,
                    bucket=bucket,
                    key=name,
                ) from exc

    @asynchronous
    def _get_object_info(
        self,
        bucket: str,
        name: str,
    ) -> tuple[str | None, Meta]:
        try:
            response: Any = self._s3_client.head_object(
                Bucket=bucket,
                Key=name,
            )

        except ClientError as exc:
            raise _translate_client_error(
                error=exc,
                bucket=bucket,
                key=name,
            ) from exc

        else:
            return response.get("ContentType"), Meta.of(response.get("Metadata"))

        # Satisfy the type checker; control flow always leaves in try/except/else.
        return None, META_EMPTY

    async def upload(
        self,
        uri: str,
        content: ResourceContent,
        **extra: Any,
    ) -> Meta:
        """Upload ``ResourceContent`` to an S3 location.

        Parameters
        ----------
        uri
            Destination S3 URI (``s3://bucket/key``).
        content
            Data and metadata to upload.

        Returns
        -------
        Meta
            Empty metadata placeholder; retained for interface parity
            with repository interactions.

        Raises
        ------
        ValueError
            If the ``uri`` does not use the ``s3://`` scheme.
        AWSAccessDenied
            If S3 rejects the request due to missing or invalid credentials.
        AWSResourceNotFound
            If the bucket does not exist.
        AWSError
            For other S3 client failures.
        """
        if not uri.startswith("s3://"):
            raise ValueError("Unsupported upload uri scheme")

        parsed_uri: ParseResult = urlparse(uri)  # s3://bucket/name
        bucket: str = parsed_uri.netloc
        name: str = parsed_uri.path.lstrip("/")
        await self._upload(
            bucket=bucket,
            name=name,
            content=content.to_bytes(),
            content_type=content.mime_type,
            meta=content.meta,
        )

        return META_EMPTY

    @asynchronous
    def _upload(
        self,
        bucket: str,
        name: str,
        content: bytes,
        content_type: str | None = None,
        meta: Mapping[str, BasicValue] | None = None,
    ) -> None:
        try:
            self._s3_client.put_object(
                Bucket=bucket,
                Body=content,
                Key=name,
                ContentType=content_type if content_type else None,
                Metadata=_sanitize_metadata(meta),
            )

        except ClientError as exc:
            raise _translate_client_error(
                error=exc,
                bucket=bucket,
                key=name,
            ) from exc


def _translate_client_error(
    *,
    error: ClientError,
    bucket: str,
    key: str,
) -> Exception:
    # type: ignore[reportAttributeAccessIssue]
    error_info: Mapping[str, Any] = getattr(error, "response", {}).get("Error", {})
    # type: ignore[reportAttributeAccessIssue]
    response_metadata: Mapping[str, Any] = getattr(error, "response", {}).get(
        "ResponseMetadata",
        {},
    )

    code: str = str(error_info.get("Code") or "").strip()
    message: str = str(error_info.get("Message") or str(error)).strip()
    status_code = response_metadata.get("HTTPStatusCode")
    uri: str = f"s3://{bucket}/{key}"

    normalized_code: str = code.lower()
    if normalized_code in {"nosuchkey", "notfound", "404"} or status_code == 404:  # noqa: PLR2004
        return AWSResourceNotFound(
            uri=uri,
            code=code or None,
            message=message,
        )

    if normalized_code in {
        "accessdenied",
        "accessdeniedexception",
        "invalidaccesskeyid",
        "signaturedoesnotmatch",
    } or status_code in {401, 403}:
        return AWSAccessDenied(
            uri=uri,
            code=code or None,
            message=message,
        )

    if normalized_code == "nosuchbucket":
        return AWSResourceNotFound(
            uri=f"s3://{bucket}",
            code=code or None,
            message=message,
        )

    return AWSError(
        uri=uri,
        code=code or None,
        message=message,
    )


def _sanitize_metadata_value(value: Any) -> str:
    # Convert to string first
    text: str = str(value)

    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Replace newlines and tabs with spaces
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Remove any control characters
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # S3 metadata value limit is 1,024 bytes per value
    max_bytes = 1021  # Reserve 3 bytes for "..."
    text_bytes = text.encode("utf-8")

    if len(text_bytes) > max_bytes:
        # Truncate at byte level and ensure valid UTF-8
        truncated_bytes = text_bytes[:max_bytes]
        # Decode with 'ignore' to handle partial characters at the end
        text = truncated_bytes.decode("utf-8", "ignore") + "..."

    return text


def _sanitize_metadata(
    meta: Mapping[str, BasicValue] | None,
) -> dict[str, str]:
    if not meta:
        return {}

    sanitized: dict[str, str] = {}
    for key, value in meta.items():
        # Sanitize both key and value
        sanitized_key = _sanitize_metadata_value(key)
        sanitized_value = _sanitize_metadata_value(value)
        if sanitized_key and sanitized_value:  # Skip empty keys/values
            sanitized[sanitized_key] = sanitized_value

    return sanitized
