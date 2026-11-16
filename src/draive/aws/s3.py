import mimetypes
import os
import re
import stat
from asyncio import gather
from collections.abc import Collection, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import ParseResult, quote, unquote, urlparse
from uuid import uuid4

from botocore.exceptions import ClientError
from haiway import BasicValue, Meta, Paginated, Pagination, asynchronous

from draive.aws.api import AWSAPI
from draive.aws.types import AWSAccessDenied, AWSError, AWSInvalidArgument, AWSResourceNotFound
from draive.resources import ResourceContent, ResourceReference

__all__ = ("AWSS3Mixin",)


class AWSS3Mixin(AWSAPI):
    async def list_paginated(
        self,
        *,
        pagination: Pagination | None = None,
        uri: str | None = None,
        recursive: bool = True,
        **extra: Any,
    ) -> Paginated[ResourceReference]:
        pagination = pagination if pagination is not None else Pagination.of(limit=32)

        if uri is None:
            references: Sequence[ResourceReference] = await self._list_buckets(
                limit=None,
                **extra,
            )
            start: int = 0
            if pagination.token is not None:
                bucket_token: str = pagination.token if isinstance(pagination.token, str) else ""
                if not bucket_token.startswith("buckets:"):
                    raise ValueError("Invalid buckets pagination token")

                start = int(bucket_token.split(":", 1)[1])

            end: int = max(start, 0) + pagination.limit
            next_token: str | None
            if end < len(references):
                next_token = f"buckets:{end}"

            else:
                next_token = None

            return Paginated[ResourceReference](
                references[start:end],
                pagination=pagination.with_token(next_token),
            )

        if not uri.startswith("s3://"):
            raise ValueError("Unsupported list uri scheme")

        bucket, prefix = _parse_s3_uri(uri)  # s3://bucket/prefix
        references, token = await self._list_page(
            bucket=bucket,
            prefix=prefix,
            recursive=recursive,
            pagination=pagination,
            **extra,
        )

        return Paginated[ResourceReference](
            references,
            pagination=pagination.with_token(token),
        )

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

        bucket, prefix = _parse_s3_uri(uri)  # s3://bucket/prefix
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

        bucket, name = _parse_s3_uri(uri)  # s3://bucket/name
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

        bucket, name = _parse_s3_uri(uri)  # s3://bucket/name
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
        target: Path = Path(save_path)
        with NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".part",
            delete=False,
        ) as file:
            temporary_path: Path = Path(file.name)
            try:
                self._s3_client.download_fileobj(
                    Bucket=bucket,
                    Key=name,
                    Fileobj=file,
                )

            except BaseException as exc:
                file.close()
                temporary_path.unlink(missing_ok=True)
                if isinstance(exc, ClientError):
                    raise _translate_client_error(
                        error=exc,
                        bucket=bucket,
                        key=name,
                    ) from exc

                raise

        try:
            temporary_path.chmod(_download_mode(target))
            temporary_path.replace(target)

        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

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
            Data and metadata to upload. Metadata keys and values are sent as S3 user
            metadata, which must be ASCII: control characters are stripped, whitespace is
            collapsed, values over 1,024 bytes are truncated and empty entries are skipped.

        Returns
        -------
        Meta
            Empty metadata placeholder; retained for interface parity
            with repository interactions.

        Raises
        ------
        ValueError
            If the ``uri`` does not use the ``s3://`` scheme.
        AWSInvalidArgument
            If any metadata key or value contains non-ASCII characters. Raised before
            any request is made; the message names the offending key but not its value.
        AWSAccessDenied
            If S3 rejects the request due to missing or invalid credentials.
        AWSResourceNotFound
            If the bucket does not exist.
        AWSError
            For other S3 client failures.
        """
        if not uri.startswith("s3://"):
            raise ValueError("Unsupported upload uri scheme")

        bucket, name = _parse_s3_uri(uri)  # s3://bucket/name
        await self._upload(
            bucket=bucket,
            name=name,
            content=content.to_bytes(),
            content_type=content.mime_type,
            meta=content.meta,
        )

        return Meta.empty

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
        if limit is not None and limit <= 0:
            return ()  # S3 rejects zero page sizes, there is nothing to request

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
                            _s3_uri(bucket, prefix_key),
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
    def _list_page(
        self,
        *,
        bucket: str,
        prefix: str,
        recursive: bool,
        pagination: Pagination,
        **extra: Any,
    ) -> tuple[Sequence[ResourceReference], str | None]:
        try:
            request: dict[str, Any] = {
                "Bucket": bucket,
                "Prefix": prefix,
                "MaxKeys": min(1000, pagination.limit),
            }
            if pagination.token is not None:
                request["ContinuationToken"] = pagination.token

            if not recursive:
                request["Delimiter"] = "/"

            if extra:
                request.update(extra)

            response: Mapping[str, Any] = self._s3_client.list_objects_v2(**request)

            references: list[ResourceReference] = []
            for entry in response.get("Contents", []):
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

            for common_prefix in response.get("CommonPrefixes", []):
                prefix_key = common_prefix.get("Prefix", "")
                if not prefix_key:
                    continue

                references.append(
                    ResourceReference.of(
                        _s3_uri(bucket, prefix_key),
                        name=_object_name(prefix_key),
                        meta=Meta.of({"type": "prefix"}),
                    )
                )

            return references, response.get("NextContinuationToken")

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
            raise _translate_client_error(
                error=exc,
                bucket="",
                key="",
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
        # botocore rejects explicit `None` instead of skipping it, the key has to be absent
        parameters: dict[str, Any] = {
            "Bucket": bucket,
            "Body": content,
            "Key": name,
            "Metadata": _sanitize_metadata(meta, uri=_s3_uri(bucket, name)),
        }
        if content_type:
            parameters["ContentType"] = content_type

        try:
            self._s3_client.put_object(**parameters)

        except ClientError as exc:
            raise _translate_client_error(
                error=exc,
                bucket=bucket,
                key=name,
            ) from exc


def _download_mode(
    target: Path,
    /,
) -> int:
    # temporary files are created owner-only, final file should look like any other new one
    try:
        return stat.S_IMODE(target.stat().st_mode)  # keep the mode of the replaced file

    except FileNotFoundError:
        pass

    # a probe file picks the umask default without touching the process-wide mask,
    # which can't be read without changing it and would race with other threads
    probe: str = str(target.parent / f".{target.name}.{uuid4().hex}.mode")
    descriptor: int = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        return stat.S_IMODE(os.fstat(descriptor).st_mode)

    finally:
        os.close(descriptor)
        os.unlink(probe)


def _sanitize_metadata_text(
    value: Any,
    /,
    *,
    key: str,
    uri: str,
    subject: str,
) -> str:
    # Convert to string first
    text: str = str(value)

    # Replace newlines and tabs with spaces
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Remove any control characters
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # S3 user metadata travels in HTTP headers and botocore rejects anything but ASCII
    if not text.isascii():
        raise AWSInvalidArgument(
            uri=uri,
            message=(
                f"S3 metadata {subject} for key {key!r} contains non-ASCII characters,"
                " encode it (e.g. percent-encoding) before passing it as metadata"
            ),
        )

    # S3 metadata value limit is 1,024 bytes per value
    max_bytes = 1021  # Reserve 3 bytes for "..."
    if len(text) > max_bytes:
        text = text[:max_bytes] + "..."

    return text


def _sanitize_metadata(
    meta: Mapping[str, BasicValue] | None,
    *,
    uri: str = "",
) -> dict[str, str]:
    if not meta:
        return {}

    sanitized: dict[str, str] = {}
    for key, value in meta.items():
        # Sanitize both key and value
        sanitized_key = _sanitize_metadata_text(key, key=str(key), uri=uri, subject="key")
        sanitized_value = _sanitize_metadata_text(
            value,
            key=str(key),
            uri=uri,
            subject="value",
        )
        if sanitized_key and sanitized_value:  # Skip empty keys/values
            sanitized[sanitized_key] = sanitized_value

    return sanitized


def _s3_uri(
    bucket: str,
    key: str,
) -> str:
    if not key:
        return f"s3://{bucket}"

    return f"s3://{bucket}/{quote(key, safe='/')}"


def _parse_s3_uri(
    uri: str,
) -> tuple[str, str]:
    parsed_uri: ParseResult = urlparse(uri)  # s3://bucket/key
    return parsed_uri.netloc, unquote(parsed_uri.path.lstrip("/"))


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
        _s3_uri(bucket, key),
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
    error_info: Mapping[str, Any] = getattr(error, "response", {}).get("Error", {})
    response_metadata: Mapping[str, Any] = getattr(error, "response", {}).get(
        "ResponseMetadata",
        {},
    )

    code: str = str(error_info.get("Code") or "").strip()
    message: str = str(error_info.get("Message") or str(error)).strip()
    status_code = response_metadata.get("HTTPStatusCode")
    uri: str = _s3_uri(bucket, key)

    normalized_code: str = code.lower()
    if normalized_code == "nosuchbucket":
        return AWSResourceNotFound(
            uri=f"s3://{bucket}",
            code=code or None,
            message=message,
        )

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

    return AWSError(
        uri=uri,
        code=code or None,
        message=message,
    )
