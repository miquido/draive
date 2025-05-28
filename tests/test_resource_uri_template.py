import json

from pytest import mark

from draive.resources import ResourceContent, resource


def test_simple_variable_expansion():
    @resource(uri_template="https://api.example.com/users/{user_id}")
    async def template(user_id: str) -> ResourceContent:
        return ResourceContent(
            blob=json.dumps({"user_id": user_id}).encode(),
            mime_type="application/json",
        )

    params = template._extract_uri_parameters("https://api.example.com/users/123")

    assert params == {"user_id": "123"}


def test_query_parameter_extraction():
    @resource(uri_template="https://api.example.com/search{?q,limit}")
    async def template(q: str, limit: str) -> ResourceContent:
        return ResourceContent(
            blob=json.dumps({"query": q, "limit": limit}).encode(),
            mime_type="application/json",
        )

    params = template._extract_uri_parameters("https://api.example.com/search?q=python&limit=10")

    assert "q" in params
    assert "limit" in params


def test_multiple_path_parameters():
    @resource(uri_template="https://api.example.com/{org}/{repo}/issues/{issue_id}")
    async def template(org: str, repo: str, issue_id: str) -> ResourceContent:
        return ResourceContent(
            blob=json.dumps({"org": org, "repo": repo, "issue_id": issue_id}).encode(),
            mime_type="application/json",
        )

    params = template._extract_uri_parameters(
        "https://api.example.com/octocat/hello-world/issues/42",
    )

    assert params == {"org": "octocat", "repo": "hello-world", "issue_id": "42"}


def test_no_parameters():
    @resource(uri_template="https://api.example.com/status")
    async def template() -> ResourceContent:
        return ResourceContent(
            blob=json.dumps({"status": "ok"}).encode(),
            mime_type="application/json",
        )

    params = template._extract_uri_parameters("https://api.example.com/status")

    assert params == {}


def test_mixed_parameters():
    @resource(uri_template="https://api.example.com/users/{user_id}/posts")
    async def template(user_id: str, page: str, limit: str) -> ResourceContent:
        return ResourceContent(
            blob=json.dumps({"user_id": user_id, "page": page, "limit": limit}).encode(),
            mime_type="application/json",
        )

    params = template._extract_uri_parameters(
        "https://api.example.com/users/123/posts?page=2&limit=10",
    )

    assert params["user_id"] == "123"
    assert params["page"] == "2"
    assert params["limit"] == "10"


@mark.asyncio
async def test_resolve_from_uri_with_parameters():
    @resource(uri_template="https://api.example.com/users/{user_id}")
    async def get_user(user_id: str):
        return ResourceContent.of(f"User {user_id} data")

    resource_result = await get_user.resolve_from_uri("https://api.example.com/users/123")

    assert resource_result.uri == "https://api.example.com/users/123"
    assert resource_result.name == "get_user"
    assert isinstance(resource_result.content, ResourceContent)


@mark.asyncio
async def test_resolve_from_uri_no_parameters():
    @resource(uri_template="https://api.example.com/status")
    async def get_status():
        return ResourceContent.of("Service is healthy")

    resource_result = await get_status.resolve_from_uri("https://api.example.com/status")

    assert resource_result.uri == "https://api.example.com/status"
    assert resource_result.name == "get_status"
    assert isinstance(resource_result.content, ResourceContent)


@mark.asyncio
async def test_resolve_from_uri_complex_template():
    @resource(uri_template="https://api.example.com/{org}/{repo}/issues/{issue_id}")
    async def get_issue(org: str, repo: str, issue_id: str):
        return ResourceContent.of(f"Issue {issue_id} in {org}/{repo}")

    resource_result = await get_issue.resolve_from_uri(
        "https://api.example.com/octocat/hello-world/issues/42"
    )

    assert resource_result.uri == "https://api.example.com/octocat/hello-world/issues/42"
    assert resource_result.name == "get_issue"
    assert isinstance(resource_result.content, ResourceContent)


@mark.asyncio
async def test_resolve_from_uri_with_query_params():
    @resource(uri_template="https://api.example.com/search")
    async def search(q: str, limit: int):
        return ResourceContent.of(f"Search results for '{q}' with limit {limit}")

    resource_result = await search.resolve_from_uri(
        "https://api.example.com/search?q=python&limit=20"
    )

    assert resource_result.uri == "https://api.example.com/search?q=python&limit=20"
    assert resource_result.name == "search"
    assert isinstance(resource_result.content, ResourceContent)


@mark.asyncio
async def test_resolve_from_uri_with_missing_default_params():
    @resource(uri_template="https://api.example.com/search")
    async def search(q: str, limit: int = 10):
        return ResourceContent.of(f"Search results for '{q}' with limit {limit}")

    resource_result = await search.resolve_from_uri("https://api.example.com/search?q=python")

    assert resource_result.uri == "https://api.example.com/search?q=python"
    assert resource_result.name == "search"
    assert isinstance(resource_result.content, ResourceContent)


def test_path_segment_expansion():
    @resource(uri_template="https://api.example.com{/resource}/{resource_id}")
    async def template(resource: str, resource_id: int) -> ResourceContent:
        return ResourceContent(
            blob=json.dumps({"resource": resource, "id": resource_id}).encode(),
            mime_type="application/json",
        )

    params = template._extract_uri_parameters("https://api.example.com/users/123")

    assert params == {"resource": "users", "resource_id": "123"}


def test_fragment_expansion():
    @resource(uri_template="https://api.example.com/doc{#section}")
    async def template(section: str) -> ResourceContent:
        return ResourceContent(
            blob=json.dumps({"section": section}).encode(),
            mime_type="application/json",
        )

    params = template._extract_uri_parameters("https://api.example.com/doc#introduction")

    assert params == {"section": "introduction"}


def test_query_template_expansion():
    @resource(uri_template="https://api.example.com/search{?q,limit}")
    async def template(q: str, limit: int) -> ResourceContent:
        return ResourceContent(
            blob=json.dumps({"query": q, "limit": limit}).encode(),
            mime_type="application/json",
        )

    params = template._extract_uri_parameters("https://api.example.com/search?q=test&limit=5")

    assert params == {"q": "test", "limit": "5"}


def test_template_parameter_validation():
    @resource(uri_template="https://api.example.com/users/{user_id}")
    async def valid_template(user_id: str) -> ResourceContent:
        return ResourceContent.of("test")

    # This should fail - template has param not in function
    try:

        @resource(uri_template="https://api.example.com/users/{user_id}/posts/{post_id}")
        async def invalid_template(user_id: str) -> ResourceContent:  # missing post_id
            return ResourceContent.of("test")

        raise AssertionError("Should have raised ValueError")

    except ValueError as e:
        assert "post_id" in str(e)


def test_template_matching_complex_template():
    @resource(uri_template="https://api.example.com/{org}/{repo}/issues{/issue_id}")
    async def template(org: str, repo: str, issue_id: str):
        return ResourceContent.of(f"Issue {issue_id} in {org}/{repo}")

    assert template.matches_uri("https://api.example.com/octocat/hello-world/issues/42")
    assert not template.matches_uri("https://api.example.com/octocat/hello-world/issues")
