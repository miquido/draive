try:
    import boto3  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "draive.aws requires the 'aws' extra. Install via `pip install draive[aws]`."
    ) from exc

from draive.aws.client import AWS
from draive.aws.observability import CloudwatchObservability
from draive.aws.state import AWSSQS, AWSCloudwatch
from draive.aws.types import AWSAccessDenied, AWSError, AWSResourceNotFound

__all__ = (
    "AWS",
    "AWSSQS",
    "AWSAccessDenied",
    "AWSCloudwatch",
    "AWSError",
    "AWSResourceNotFound",
    "CloudwatchObservability",
)
