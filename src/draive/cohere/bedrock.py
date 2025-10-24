import base64
import json
import re
import typing

import httpx
from cohere import (
    ApiMeta,
    ApiMetaBilledUnits,
    ApiMetaTokens,
    EmbedResponse,
    GenerateStreamedResponse,
    Generation,
    NonStreamedChatResponse,
    RerankResponse,
    StreamedChatResponse,
)
from cohere.client import ClientEnvironment  # pyright: ignore
from cohere.client_v2 import AsyncClientV2
from cohere.core import construct_type
from cohere.manually_maintained.lazy_aws_deps import lazy_boto3, lazy_botocore
from haiway import asynchronous
from httpx import URL, AsyncByteStream, ByteStream

__all__ = ("AsyncBedrockClientV2",)


# it is a naive adjustment of cohere.BedrockClientV2 to work with async interfaces
class AsyncAwsClientV2(AsyncClientV2):
    def __init__(
        self,
        *,
        aws_region: str | None = None,
        service: typing.Literal["bedrock"] | typing.Literal["sagemaker"],
    ):
        AsyncClientV2.__init__(
            self,
            base_url="https://api.cohere.com",  # this url is unused for BedrockClient
            environment=ClientEnvironment.PRODUCTION,
            client_name="n/a",
            api_key="n/a",
            httpx_client=httpx.AsyncClient(
                event_hooks=get_event_hooks(
                    service=service,
                    aws_region=aws_region,
                ),
            ),
        )


class AsyncBedrockClientV2(AsyncAwsClientV2):
    def __init__(
        self,
        *,
        aws_region: str | None = None,
    ):
        AsyncAwsClientV2.__init__(
            self,
            service="bedrock",
            aws_region=aws_region,
        )


EventHook = typing.Callable[..., typing.Any]


def get_event_hooks(
    service: str,
    aws_region: str | None,
) -> dict[str, list[EventHook]]:
    return {
        "request": [
            map_request_to_bedrock(
                service=service,
                aws_region=aws_region,
            ),
        ],
        "response": [map_response_from_bedrock()],
    }


class TextGeneration(typing.TypedDict):
    text: str
    is_finished: str
    event_type: typing.Literal["text-generation"]


class StreamEnd(typing.TypedDict):
    is_finished: str
    event_type: typing.Literal["stream-end"]
    finish_reason: str


class Streamer(AsyncByteStream):
    lines: typing.AsyncIterator[bytes]

    def __init__(self, lines: typing.AsyncIterator[bytes]):
        self.lines = lines

    def __aiter__(self) -> typing.AsyncIterator[bytes]:
        return self.lines


response_mapping: dict[str, typing.Any] = {
    "chat": NonStreamedChatResponse,
    "embed": EmbedResponse,
    "generate": Generation,
    "rerank": RerankResponse,
}

stream_response_mapping: dict[str, typing.Any] = {
    "chat": StreamedChatResponse,
    "generate": GenerateStreamedResponse,
}


async def stream_generator(response: httpx.Response, endpoint: str) -> typing.AsyncIterator[bytes]:
    regex = r"{[^\}]*}"

    async for _text in response.aiter_lines():
        match = re.search(regex, _text)
        if match:
            obj = json.loads(match.group())
            if "bytes" in obj:
                base64_payload = base64.b64decode(obj["bytes"]).decode("utf-8")
                streamed_obj = json.loads(base64_payload)
                if "event_type" in streamed_obj:
                    response_type = stream_response_mapping[endpoint]
                    parsed = typing.cast(  # type: ignore
                        response_type,  # type: ignore
                        construct_type(type_=response_type, object_=streamed_obj),
                    )
                    yield (json.dumps(parsed.dict()) + "\n").encode("utf-8")  # type: ignore


def map_token_counts(response: httpx.Response) -> ApiMeta:
    input_tokens = int(response.headers.get("X-Amzn-Bedrock-Input-Token-Count", -1))
    output_tokens = int(response.headers.get("X-Amzn-Bedrock-Output-Token-Count", -1))
    return ApiMeta(
        tokens=ApiMetaTokens(input_tokens=input_tokens, output_tokens=output_tokens),
        billed_units=ApiMetaBilledUnits(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def map_response_from_bedrock():
    async def _hook(
        response: httpx.Response,
    ) -> None:
        stream = response.headers["content-type"] == "application/vnd.amazon.eventstream"
        endpoint = response.request.extensions["endpoint"]
        output: typing.AsyncIterator[bytes]

        if stream:
            output = stream_generator(
                httpx.Response(
                    stream=response.stream,
                    status_code=response.status_code,
                ),
                endpoint,
            )

        else:
            response_type = response_mapping[endpoint]
            response_obj = json.loads(await response.aread())
            response_obj["meta"] = map_token_counts(response).dict()
            cast_obj: typing.Any = typing.cast(  # type: ignore
                response_type,  # type: ignore
                construct_type(
                    type_=response_type,
                    # type: ignore
                    object_=response_obj,
                ),
            )

            async def output_generator() -> typing.AsyncIterator[bytes]:
                yield json.dumps(cast_obj.dict()).encode("utf-8")  # type: ignore

            output = output_generator()

        response.stream = Streamer(output)

        # reset response object to allow for re-reading
        if hasattr(response, "_content"):
            del response._content  # type: ignore
        response.is_stream_consumed = False
        response.is_closed = False

    return _hook


def map_request_to_bedrock(
    service: str,
    aws_region: str | None,
) -> EventHook:
    session = lazy_boto3().Session(region_name=aws_region)
    aws_region = session.region_name  # type: ignore
    credentials = session.get_credentials()  # type: ignore
    signer = lazy_botocore().auth.SigV4Auth(credentials, service, aws_region)  # pyright: ignore

    @asynchronous
    def _event_hook(request: httpx.Request) -> None:
        headers = request.headers.copy()
        del headers["connection"]

        api_version = request.url.path.split("/")[-2]
        endpoint = request.url.path.split("/")[-1]
        body = json.loads(request.read())
        model = body["model"]

        url = get_url(
            platform=service,
            aws_region=aws_region,  # type: ignore
            model=model,  # type: ignore
            stream="stream" in body and body["stream"],
        )
        request.url = URL(url)
        request.headers["host"] = request.url.host

        if endpoint == "rerank":
            body["api_version"] = get_api_version(version=api_version)

        if "stream" in body:
            del body["stream"]

        if "model" in body:
            del body["model"]

        new_body = json.dumps(body).encode("utf-8")
        request.stream = ByteStream(new_body)
        request._content = new_body  # type: ignore
        headers["content-length"] = str(len(new_body))

        aws_request = lazy_botocore().awsrequest.AWSRequest(  # pyright: ignore
            method=request.method,
            url=url,
            headers=headers,
            data=request.read(),
        )
        signer.add_auth(aws_request)  # type: ignore

        request.headers = httpx.Headers(aws_request.prepare().headers)  # type: ignore
        request.extensions["endpoint"] = endpoint

    return _event_hook


def get_url(
    *,
    platform: str,
    aws_region: str | None,
    model: str,
    stream: bool,
) -> str:
    if platform == "bedrock":
        endpoint = "invoke" if not stream else "invoke-with-response-stream"
        return f"https://{platform}-runtime.{aws_region}.amazonaws.com/model/{model}/{endpoint}"
    elif platform == "sagemaker":
        endpoint = "invocations" if not stream else "invocations-response-stream"
        return f"https://runtime.sagemaker.{aws_region}.amazonaws.com/endpoints/{model}/{endpoint}"
    return ""


def get_api_version(*, version: str):
    int_version = {
        "v1": 1,
        "v2": 2,
    }

    return int_version.get(version, 1)
