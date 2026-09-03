from draive.anthropic.api import AnthropicAPI


def test_bedrock_client_targets_bedrock_runtime() -> None:
    client = AnthropicAPI(
        "bedrock",
        aws_region="us-east-1",
    )._prepare_client()  # pyright: ignore[reportPrivateUsage]

    # the marketplace client requires a workspace and can't be used for bedrock
    assert client.base_url.host == "bedrock-runtime.us-east-1.amazonaws.com"
    assert client.max_retries == 0  # retries are handled within draive
    assert hasattr(client.messages, "stream")
