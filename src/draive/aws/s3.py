import mimetypes
import re
from asyncio import gather
from collections.abc import Collection, Mapping, Sequence
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
    async def list(
        self,
        *,
        uri: str | None = None,
        recursive: bool = True,
        limit: int | None = None,
        **extra: Any,
    ) -> Sequence[ResourceReference]:
        """List buckets or objects stored under an S3 bucket or prefix.

        Parameters
        ----------
        uri
            Optional S3 URI pointing to the bucket or prefix
            (``s3://bucket/prefix``). When omitted, lists buckets.
        recursive
            When ``True`` (default), list all objects under the prefix.
            When ``False``, list only the immediate children and prefix markers.
        limit
            Optional maximum number of results to return.

        Returns
        -------
        Sequence[ResourceReference]
            References to objects and prefix markers under the requested scope.

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
        if uri is None:
            return await self._list_buckets(
                limit=limit,
                **extra,
            )

        if not uri.startswith("s3://"):
            raise ValueError("Unsupported list uri scheme")

        parsed_uri: ParseResult = urlparse(uri)  # s3://bucket/prefix
        bucket: str = parsed_uri.netloc
        prefix: str = parsed_uri.path.lstrip("/")
        return await self._list(
            bucket=bucket,
            prefix=prefix,
            recursive=recursive,
            limit=limit,
            **extra,
        )

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
    def _list(  # noqa: C901, PLR0912
        self,
        *,
        bucket: str,
        prefix: str,
        recursive: bool,
        limit: int | None,
        **extra: Any,
    ) -> Sequence[ResourceReference]:
        try:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            pagination_config: dict[str, int] = {}
            if limit is not None:
                pagination_config["MaxItems"] = limit
                pagination_config["PageSize"] = min(1000, limit)

            request: dict[str, Any] = {
                "Bucket": bucket,
                "Prefix": prefix,
            }
            if not recursive:
                request["Delimiter"] = "/"

            if pagination_config:
                request["PaginationConfig"] = pagination_config

            if extra:
                request.update(extra)

            references: list[ResourceReference] = []
            for page in paginator.paginate(**request):
                for entry in page.get("Contents", []):
                    if limit is not None and len(references) >= limit:
                        break

                    key = entry.get("Key", "")
                    if not key or (key == prefix and key.endswith("/")):
                        continue

                    references.append(
                        _object_reference(
                            bucket=bucket,
                            key=key,
                            entry=entry,
                        )
                    )

                if limit is not None and len(references) >= limit:
                    break

                for common_prefix in page.get("CommonPrefixes", []):
                    if limit is not None and len(references) >= limit:
                        break

                    prefix_key = common_prefix.get("Prefix", "")
                    if not prefix_key:
                        continue

                    references.append(
                        ResourceReference.of(
                            f"s3://{bucket}/{prefix_key}",
                            name=_object_name(prefix_key),
                            meta=Meta.of({"type": "prefix"}),
                        )
                    )

                if limit is not None and len(references) >= limit:
                    break

            return references

        except ClientError as exc:
            raise _translate_client_error(
                error=exc,
                bucket=bucket,
                key=prefix,
            ) from exc

    @asynchronous
    def _list_buckets(
        self,
        *,
        limit: int | None,
        **extra: Any,
    ) -> Sequence[ResourceReference]:
        try:
            response: Mapping[str, Any] = self._s3_client.list_buckets(**extra)
            buckets: Collection[Mapping[str, Any]] = response.get("Buckets", ())
            references: list[ResourceReference] = [
                _bucket_reference(bucket) for bucket in buckets if bucket.get("Name")
            ]
            if limit is not None:
                return references[:limit]

            return references

        except ClientError as exc:
            error_info: Mapping[str, Any] = getattr(exc, "response", {}).get("Error", {})
            code: str = str(error_info.get("Code") or "").strip()
            message: str = str(error_info.get("Message") or str(exc)).strip()
            raise AWSError(
                uri="s3://",
                code=code or None,
                message=message,
            ) from exc

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


def _sanitize_metadata_value(value: Any) -> str:
    # Convert to string first
    text: str = str(value)

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


def _object_name(key: str) -> str | None:
    trimmed = key.rstrip("/")
    if not trimmed:
        return None
    return Path(trimmed).name


def _object_reference(
    *,
    bucket: str,
    key: str,
    entry: Mapping[str, Any],
) -> ResourceReference:
    meta_values: dict[str, BasicValue] = {}
    meta_values["type"] = "object"
    if (etag := entry.get("ETag")) is not None:
        meta_values["etag"] = str(etag).strip('"')
    if (size := entry.get("Size")) is not None:
        meta_values["size"] = int(size)
    if (last_modified := entry.get("LastModified")) is not None:
        meta_values["last_modified"] = str(last_modified)
    if (storage_class := entry.get("StorageClass")) is not None:
        meta_values["storage_class"] = str(storage_class)
    if (mime_type := mimetypes.guess_type(key)[0]) is not None:
        meta_values["mime_type"] = mime_type

    return ResourceReference.of(
        f"s3://{bucket}/{key}",
        name=_object_name(key),
        meta=Meta.of(meta_values),
    )


def _bucket_reference(
    bucket: Mapping[str, Any],
) -> ResourceReference:
    name = str(bucket.get("Name") or "").strip()
    meta_values: dict[str, BasicValue] = {}
    meta_values["type"] = "bucket"
    if creation_date := bucket.get("CreationDate"):
        meta_values["creation_date"] = str(creation_date)

    return ResourceReference.of(
        f"s3://{name}",
        name=name or None,
        meta=Meta.of(meta_values),
    )


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
