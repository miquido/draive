import asyncio
import json
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from haiway import ValidationError, ctx
from haiway.context.types import ContextStateMissing

from draive import (
    Agent,
    AgentMemory,
    ModelInput,
    ModelOutput,
    ResourceContent,
    ResourceReference,
    Template,
    TemplatesRepository,
    TextContent,
)
from draive.claude import (
    ClaudeCode,
    ClaudeCodeApiKey,
    ClaudeCodeConfig,
    ClaudeCodeEnded,
    ClaudeCodeErrored,
    ClaudeCodeFailed,
    ClaudeCodeOAuthToken,
    ClaudeCodeResult,
    ClaudeCodeRunner,
    ClaudeCodeSessions,
    ClaudeCodeStarted,
    ClaudeCodeSuccess,
    ClaudeCodeTelemetry,
    ClaudeCodeUnavailable,
    claude_code_agent,
)
from draive.claude.sessions import session_exists, session_identifier
from draive.multimodal import ArtifactContent, MultimodalContent
from draive.utils import ProcessingEvent
from tests._claude_cli import alive, write_fake_claude

SECRET = "sk-ant-api03-secret-credential"


class _Setup:
    def __init__(
        self,
        root: Path,
        *,
        sessions: ClaudeCodeSessions | None = None,
        **config: Any,
    ) -> None:
        self.sessions = sessions
        self.binary = write_fake_claude(root)
        self.record = root / "record.json"
        scenario = config.pop("scenario", "echo")
        env = config.pop("env", {})
        self.config = ClaudeCodeConfig(
            timeout=30,
            stop_grace=2,
            env={
                "FAKE_CLAUDE_SCENARIO": scenario,
                "FAKE_CLAUDE_RECORD": str(self.record),
                "FAKE_CLAUDE_CHILD": str(root / "child.pid"),
                **env,
            },
            **config,
        )

    def scope(self, *state: Any) -> Any:
        return ctx.scope(
            "test",
            self.config,
            *state,
            disposables=(
                ClaudeCode(binary=self.binary, credential=ClaudeCodeApiKey(value=SECRET)),
            ),
        )

    def agent(self, name: str, **arguments: Any) -> Agent:
        """An agent of these sessions when there are any."""
        if self.sessions is not None:
            arguments["memory"] = self.sessions

        return claude_code_agent(name, **arguments)

    def recorded(self) -> Any:
        return json.loads(self.record.read_text())


async def _call(agent: Any, text: Any = "hello", thread: UUID | None = None) -> list[Any]:
    return [chunk async for chunk in agent.call(input=text, thread=thread)]


def _texts(chunks: list[Any]) -> list[str]:
    return [chunk.text for chunk in chunks if isinstance(chunk, TextContent)]


def _events(chunks: list[Any]) -> list[ProcessingEvent]:
    return [chunk for chunk in chunks if isinstance(chunk, ProcessingEvent)]


@pytest.mark.asyncio
async def test_agent_streams_text_and_reports_tools_and_result(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, scenario="tools")
    agent = claude_code_agent("coder", description="codes")

    async with setup.scope():
        chunks = await _call(agent)

    assert _texts(chunks) == ["reading", "done"]
    use, result_of_tool, summary = _events(chunks)
    assert (use.event, use.meta.get("tool"), use.meta.get("identifier")) == (
        "claude_code.tool_use",
        "Read",
        "t1",
    )
    assert json.loads(str(use.meta.get("arguments"))) == {"path": "a.txt"}
    assert result_of_tool.event == "claude_code.tool_result"
    assert result_of_tool.content.to_str() == "file body"
    assert result_of_tool.meta.get("failed") is False
    assert summary.event == "claude_code.result"
    assert summary.meta.get("cost_usd") == 0.5
    assert summary.meta.get("turns") == 1
    assert chunks[-1] is summary


@pytest.mark.asyncio
async def test_agent_runs_isolated_with_its_instructions(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, model="opus", tools=("Read",))
    agent = claude_code_agent("coder", instructions="be careful")

    async with setup.scope():
        await _call(agent)

    recorded = setup.recorded()
    arguments = recorded["arguments"]
    assert arguments[arguments.index("--append-system-prompt") + 1] == "be careful"
    assert arguments[arguments.index("--model") + 1] == "opus"
    assert arguments[arguments.index("--tools") + 1] == "Read"
    assert recorded["key"] == SECRET
    assert SECRET not in json.dumps(arguments)


@pytest.mark.asyncio
async def test_template_instructions_are_resolved(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)
    agent = claude_code_agent("coder", instructions=Template.of("coder_rules"))

    async with setup.scope(TemplatesRepository.volatile(coder_rules="templated rules")):
        await _call(agent)

    arguments = setup.recorded()["arguments"]
    assert arguments[arguments.index("--append-system-prompt") + 1] == "templated rules"


@pytest.mark.asyncio
async def test_the_configuration_of_the_agent_wins_over_the_scope(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, model="scope-model")
    agent = claude_code_agent("coder", config=setup.config.updating(model="agent-model"))

    async with setup.scope():
        await _call(agent)

    arguments = setup.recorded()["arguments"]
    assert arguments[arguments.index("--model") + 1] == "agent-model"


@pytest.mark.asyncio
async def test_streamed_text_is_not_repeated_by_the_complete_block(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, scenario="deltas", stream_deltas=True)

    async with setup.scope():
        chunks = await _call(claude_code_agent("coder"))

    assert _texts(chunks) == ["hel", "lo"]
    assert "--include-partial-messages" in setup.recorded()["arguments"]


@pytest.mark.asyncio
async def test_failed_runs_raise_with_the_reason_reported_by_the_run(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, scenario="error401")

    async with setup.scope():
        with pytest.raises(ClaudeCodeErrored, match="Invalid API key") as raised:
            await _call(claude_code_agent("coder"))

    assert raised.value.api_error_status == 401
    assert raised.value.terminal_reason == "api_error"
    assert raised.value.session_id


@pytest.mark.asyncio
async def test_a_failed_run_is_not_remembered(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, scenario="error401")
    memory = AgentMemory.volatile()
    agent = claude_code_agent("coder", memory=memory)
    thread = uuid4()

    async with setup.scope():
        with pytest.raises(ClaudeCodeFailed):
            await _call(agent, "hello", thread)

        setup.config = setup.config.updating(
            env={**setup.config.env, "FAKE_CLAUDE_SCENARIO": "echo"}
        )

    async with setup.scope():
        await _call(agent, "again", thread)

    assert "hello" not in setup.recorded()["text"]


@pytest.mark.asyncio
async def test_memory_of_draive_hands_the_history_to_a_fresh_session(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)
    memory = AgentMemory.volatile()
    agent = claude_code_agent("coder", memory=memory)
    thread = uuid4()

    async with setup.scope():
        await _call(agent, "first question", thread)
        first = setup.recorded()
        assert "first question" == first["text"]  # nothing to recall yet
        assert "--no-session-persistence" in first["arguments"]

        await _call(agent, "second question", thread)
        second = setup.recorded()
        assert "--no-session-persistence" in second["arguments"]
        assert "--resume" not in second["arguments"]
        assert "<user>first question</user>" in second["text"]
        assert "<assistant>echo: first question</assistant>" in second["text"]
        assert second["text"].endswith("second question")

        # another thread has its own history
        await _call(agent, "unrelated", uuid4())
        assert setup.recorded()["text"] == "unrelated"


@pytest.mark.asyncio
async def test_remembered_turns_are_available_to_any_memory(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)
    memory = AgentMemory.volatile()
    agent = claude_code_agent("coder", memory=memory)
    thread = uuid4()
    identity_uri = agent.identity.uri

    async with setup.scope():
        await _call(agent, "remember me", thread)

        from draive.agents import AgentThread

        recalled = await memory.recall(
            thread=AgentThread.of(thread, agent_uri=identity_uri),
            context=(),
        )

    assert len(recalled) == 2
    assert isinstance(recalled[0], ModelInput)
    assert recalled[0].content.to_str() == "remember me"
    assert isinstance(recalled[1], ModelOutput)
    assert recalled[1].content.to_str() == "echo: remember me"


@pytest.mark.asyncio
async def test_resumable_sessions_continue_the_session_of_the_thread(tmp_path: Path) -> None:
    state = tmp_path / "state"
    setup = _Setup(tmp_path, sessions=ClaudeCodeSessions(state_directory=str(state)))
    agent = setup.agent("coder")
    thread = uuid4()

    async with setup.scope():
        await _call(agent, "one", thread)
        first = setup.recorded()
        arguments = first["arguments"]
        session = session_identifier(agent.identity.uri, thread)
        assert arguments[arguments.index("--session-id") + 1] == session
        assert session != str(thread)
        assert "--no-session-persistence" not in arguments
        assert first["text"] == "one"  # no history is replayed, Claude Code has it

        await _call(agent, "two", thread)
        second = setup.recorded()
        arguments = second["arguments"]
        assert arguments[arguments.index("--resume") + 1] == session
        assert "--session-id" not in arguments
        assert second["text"] == "two"
        # the same directories every time, sessions are stored by them
        assert second["directory"] == first["directory"] == str(state / "workspace")
        assert second["environment"]["HOME"] == str(state / "home")
        assert second["environment"]["CLAUDE_CONFIG_DIR"] == str(state / "claude")

        other = uuid4()
        await _call(agent, "other", other)
        arguments = setup.recorded()["arguments"]
        assert arguments[arguments.index("--session-id") + 1] == session_identifier(
            agent.identity.uri, other
        )


@pytest.mark.asyncio
async def test_a_failed_run_keeps_its_session_for_the_next_one(tmp_path: Path) -> None:
    state = tmp_path / "state"
    setup = _Setup(
        tmp_path,
        scenario="error401",
        sessions=ClaudeCodeSessions(state_directory=str(state)),
    )
    agent = setup.agent("coder")
    thread = uuid4()

    async with setup.scope():
        with pytest.raises(ClaudeCodeFailed):
            await _call(agent, "one", thread)

    setup.config = setup.config.updating(env={**setup.config.env, "FAKE_CLAUDE_SCENARIO": "echo"})
    async with setup.scope():
        await _call(agent, "two", thread)

    arguments = setup.recorded()["arguments"]
    assert arguments[arguments.index("--resume") + 1] == session_identifier(
        agent.identity.uri, thread
    )


@pytest.mark.asyncio
async def test_ephemeral_workspaces_are_removed_after_the_run(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)

    async with setup.scope():
        await _call(claude_code_agent("coder"))

    directory = Path(setup.recorded()["directory"])
    assert directory.name == "workspace"
    assert not directory.exists()
    assert not Path(setup.recorded()["environment"]["HOME"]).exists()


@pytest.mark.asyncio
async def test_configured_directories_are_used_and_kept(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    setup = _Setup(tmp_path, directory=str(project))

    async with setup.scope():
        await _call(claude_code_agent("coder"))

    assert setup.recorded()["directory"] == str(project)
    assert project.exists()


@pytest.mark.asyncio
async def test_a_missing_configured_directory_is_refused(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, directory=str(tmp_path / "missing"))

    async with setup.scope():
        with pytest.raises(ValueError, match="existing directory"):
            await _call(claude_code_agent("coder"))


def test_invalid_configuration_can_not_reach_a_run() -> None:
    # nothing to refuse when a run starts, the configuration could not be created
    with pytest.raises(ValidationError):
        ClaudeCodeConfig(env={"ANTHROPIC_API_KEY": "sneaky"})


@pytest.mark.asyncio
async def test_a_runner_which_ends_without_a_result_fails_the_run() -> None:
    async def running(launch: Any) -> Any:
        yield ClaudeCodeStarted(session_id="s")

    async with ctx.scope(
        "test",
        ClaudeCodeConfig(),
        ClaudeCodeRunner(binary="/fake", running=running),
    ):
        with pytest.raises(ClaudeCodeEnded):
            await _call(claude_code_agent("coder"))


@pytest.mark.asyncio
async def test_telemetry_continues_the_trace_of_the_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ctx,
        "trace_context",
        lambda: {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"},
    )
    setup = _Setup(
        tmp_path,
        telemetry=ClaudeCodeTelemetry(endpoint="http://collector:4317", signals={"traces"}),
    )
    agent = claude_code_agent("coder")
    thread = uuid4()

    async with setup.scope():
        await _call(agent, "hello", thread)

    environment = setup.recorded()["environment"]
    assert environment["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert environment["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"
    assert environment["TRACEPARENT"] == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    attributes = dict(
        pair.split("=") for pair in environment["OTEL_RESOURCE_ATTRIBUTES"].split(",")
    )
    assert attributes["service.name"] == "claude-code"
    assert attributes["draive.agent"] == "coder"
    assert attributes["draive.thread"] == str(thread)


@pytest.mark.asyncio
async def test_usage_and_cost_are_recorded_in_draive_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(ctx, "record_info", lambda **kwargs: records.append(kwargs))
    setup = _Setup(tmp_path)

    async with setup.scope():
        await _call(claude_code_agent("coder"))

    metrics = {record["metric"]: record["value"] for record in records if "metric" in record}
    assert metrics["model.input_tokens"] == 10
    assert metrics["model.output_tokens"] == 5
    assert metrics["model.input_tokens.cached"] == 2
    assert metrics["agent.claude_code.cost"] == 0.5
    usage = next(record for record in records if record.get("metric") == "model.input_tokens")
    assert usage["attributes"] == {"model.provider": "claude_code", "model.name": "fake-model"}


@pytest.mark.asyncio
async def test_unknown_builtin_plugins_are_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(ctx, "log_warning", lambda message, *_, **__: warnings.append(message))
    setup = _Setup(tmp_path)
    setup.binary = write_fake_claude(
        tmp_path, plugins='[{"name": "cc-plugin-new", "path": "builtin"}]'
    )

    async with setup.scope():
        await _call(claude_code_agent("coder"))

    assert any("cc-plugin-new" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_mcp_servers_which_failed_to_connect_are_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(ctx, "log_warning", lambda message, *_, **__: warnings.append(message))
    setup = _Setup(
        tmp_path,
        env={
            "FAKE_CLAUDE_MCP": json.dumps(
                [{"name": "good", "status": "connected"}, {"name": "bad", "status": "failed"}]
            )
        },
    )

    async with setup.scope():
        await _call(claude_code_agent("coder"))

    assert any("bad" in warning and "good" not in warning for warning in warnings)


@pytest.mark.asyncio
async def test_plugins_enforced_by_the_organization_are_not_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(ctx, "log_warning", lambda message, *_, **__: warnings.append(message))
    setup = _Setup(tmp_path)
    setup.binary = write_fake_claude(
        tmp_path, plugins='[{"name": "cc-plugin-sec-default", "path": "builtin"}]'
    )

    async with setup.scope():
        await _call(claude_code_agent("coder"))

    assert not warnings


@pytest.mark.asyncio
async def test_images_are_sent_and_hidden_artifacts_are_left_out(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)
    content = MultimodalContent.of(
        "look at this",
        ResourceContent.of(b"png-bytes", mime_type="image/png"),
        ArtifactContent.of({"internal": True}, category="hidden", hidden=True),
    )

    async with setup.scope():
        await _call(claude_code_agent("coder"), content)

    blocks = json.loads(setup.recorded()["line"])["message"]["content"]
    assert [block["type"] for block in blocks] == ["text", "image"]
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert blocks[1]["source"]["type"] == "base64"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        MultimodalContent.of(
            ResourceReference.of("https://example.com/a.pdf", mime_type="application/pdf")
        ),
        MultimodalContent.of("text", ResourceContent.of(b"x", mime_type="audio/wav")),
        MultimodalContent.of(ArtifactContent.of({"visible": True}, category="shown")),
        MultimodalContent.empty,
        MultimodalContent.of("   "),
    ],
)
async def test_content_which_can_not_be_sent_is_refused(tmp_path: Path, content: Any) -> None:
    setup = _Setup(tmp_path)

    async with setup.scope():
        with pytest.raises(ValueError):
            await _call(claude_code_agent("coder"), content)

    assert not setup.record.exists()


@pytest.mark.asyncio
async def test_abandoning_the_call_stops_the_run(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, scenario="hang")

    async with setup.scope():
        stream = claude_code_agent("coder").call(input="hello")
        try:
            async for chunk in stream:
                if isinstance(chunk, TextContent):
                    break

        finally:
            await stream.aclose()

    child = int((tmp_path / "child.pid").read_text())
    deadline = time.monotonic() + 5
    while alive(child) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)

    assert not alive(child)


@pytest.mark.asyncio
async def test_the_agent_requires_claude_code_within_the_scope() -> None:
    async with ctx.scope("test", ClaudeCodeConfig()):
        with pytest.raises(ContextStateMissing):
            await _call(claude_code_agent("coder"))


@pytest.mark.asyncio
async def test_without_a_configuration_the_defaults_apply(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)
    binary = setup.binary

    async with ctx.scope(
        "test",
        ClaudeCodeConfig(env=setup.config.env),
        disposables=(ClaudeCode(binary=binary, credential=ClaudeCodeApiKey(value=SECRET)),),
    ):
        await _call(claude_code_agent("coder"))

    arguments = setup.recorded()["arguments"]
    assert arguments[arguments.index("--model") + 1] == "sonnet"
    assert arguments[arguments.index("--permission-mode") + 1] == "dontAsk"
    assert arguments[arguments.index("--tools") + 1] == "Read,Grep,Glob"


@pytest.mark.asyncio
async def test_the_agent_can_be_used_as_a_tool_of_another_one(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)
    tool = claude_code_agent("coder", description="Codes things").as_tool(
        name="code",
        description="Codes",
    )

    async with setup.scope():
        assert tool.name == "code"
        assert tool.description == "Codes"


@pytest.mark.asyncio
async def test_binary_is_resolved_and_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = write_fake_claude(tmp_path)
    monkeypatch.setenv("CLAUDE_CODE_BINARY", binary)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))

    async with ctx.scope(
        "test",
        disposables=(ClaudeCode(credential=ClaudeCodeApiKey(value=SECRET)),),
    ):
        pass  # resolved from the environment

    with pytest.raises(ClaudeCodeUnavailable, match="not found"):
        async with ctx.scope(
            "test",
            disposables=(
                ClaudeCode(
                    binary=str(tmp_path / "missing"),
                    credential=ClaudeCodeApiKey(value=SECRET),
                ),
            ),
        ):
            pass

    not_executable = tmp_path / "plain"
    not_executable.write_text("#!/bin/sh")
    with pytest.raises(ClaudeCodeUnavailable, match="not found"):
        async with ctx.scope(
            "test",
            disposables=(
                ClaudeCode(
                    binary=str(not_executable),
                    credential=ClaudeCodeApiKey(value=SECRET),
                ),
            ),
        ):
            pass


@pytest.mark.asyncio
async def test_credentials_are_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = write_fake_claude(tmp_path)
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ClaudeCodeUnavailable, match="credentials"):
        async with ctx.scope("test", disposables=(ClaudeCode(binary=binary),)):
            pass


@pytest.mark.asyncio
async def test_credentials_are_resolved_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _Setup(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-from-env")

    async with ctx.scope(
        "test",
        setup.config,
        disposables=(ClaudeCode(binary=setup.binary),),
    ):
        await _call(claude_code_agent("coder"))

    recorded = setup.recorded()
    assert recorded["key"] == "sk-ant-oat-from-env"
    assert recorded["variable"] == "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"


@pytest.mark.asyncio
async def test_gateway_endpoint_is_passed_to_the_run(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)

    async with ctx.scope(
        "test",
        setup.config,
        disposables=(
            ClaudeCode(
                binary=setup.binary,
                credential=ClaudeCodeApiKey(value=SECRET),
                base_url="https://gateway.example",
            ),
        ),
    ):
        await _call(claude_code_agent("coder"))

    assert setup.recorded()["environment"]["ANTHROPIC_BASE_URL"] == "https://gateway.example"


def test_the_client_never_shows_its_credentials() -> None:
    client = ClaudeCode(credential=ClaudeCodeApiKey(value=SECRET))

    assert SECRET not in repr(client)
    assert SECRET not in str(client)


@pytest.mark.asyncio
async def test_runs_authenticate_with_the_provided_credential_only(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)

    async with setup.scope():
        await _call(claude_code_agent("coder"))

    recorded = setup.recorded()
    # nothing but the credential of the run can authenticate it
    assert recorded["key"] == SECRET
    assert recorded["environment"]["HOME"] != str(Path.home())
    assert recorded["environment"]["CLAUDE_CONFIG_DIR"] != str(Path.home() / ".claude")
    assert "ANTHROPIC_API_KEY" not in recorded["environment"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in recorded["environment"]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["/login managed key", "apiKeyHelper", "none", ""])
async def test_a_run_on_any_other_authentication_is_stopped(
    tmp_path: Path,
    source: str,
) -> None:
    # `none` is not what an API key is reported as, the run would be a logged in one
    setup = _Setup(tmp_path, scenario="hang", env={"FAKE_CLAUDE_SOURCE": source or "-"})
    chunks: list[Any] = []

    async with setup.scope():
        with pytest.raises(ClaudeCodeUnavailable, match="instead of the provided API key"):
            async for chunk in claude_code_agent("coder").call(input="hello"):
                chunks.append(chunk)

    assert chunks == []  # nothing of the run reached the consumer
    child = int((tmp_path / "child.pid").read_text())
    deadline = time.monotonic() + 5
    while alive(child) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)

    assert not alive(child)


@pytest.mark.asyncio
async def test_oauth_tokens_are_verified_by_their_own_source(tmp_path: Path) -> None:
    setup = _Setup(tmp_path)

    async with ctx.scope(
        "test",
        setup.config,
        disposables=(
            ClaudeCode(
                binary=setup.binary,
                credential=ClaudeCodeOAuthToken(value="sk-ant-oat01-token"),
            ),
        ),
    ):
        assert _texts(await _call(claude_code_agent("coder"))) == ["echo: hello"]

    assert setup.recorded()["variable"] == "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"

    setup = _Setup(tmp_path, env={"FAKE_CLAUDE_SOURCE": "ANTHROPIC_API_KEY"})
    async with ctx.scope(
        "test",
        setup.config,
        disposables=(
            ClaudeCode(
                binary=setup.binary,
                credential=ClaudeCodeOAuthToken(value="sk-ant-oat01-token"),
            ),
        ),
    ):
        with pytest.raises(ClaudeCodeUnavailable, match="instead of the provided OAuth token"):
            await _call(claude_code_agent("coder"))


@pytest.mark.parametrize(
    ("credential", "message"),
    [
        (ClaudeCodeApiKey(value="sk-ant-oat01-token"), "OAuth token"),
        (ClaudeCodeOAuthToken(value="sk-ant-api03-key"), "API key"),
    ],
)
def test_credentials_of_the_wrong_kind_are_refused(credential: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ClaudeCode(credential=credential)


def test_gateways_may_use_any_credential() -> None:
    ClaudeCode(
        credential=ClaudeCodeApiKey(value="sk-ant-oat01-token"),
        base_url="https://gateway.example",
    )
    ClaudeCode(
        credential=ClaudeCodeOAuthToken(value="gateway-token"),
        base_url="https://gateway.example",
    )


@pytest.mark.asyncio
async def test_credentials_of_the_wrong_kind_are_refused_from_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = write_fake_claude(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-oat01-token")

    with pytest.raises(ValueError, match="OAuth token"):
        async with ctx.scope("test", disposables=(ClaudeCode(binary=binary),)):
            pass


@pytest.mark.asyncio
async def test_sessions_of_different_agents_never_collide_and_survive_a_restart(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    setup = _Setup(tmp_path, sessions=ClaudeCodeSessions(state_directory=str(state)))
    thread = uuid4()

    async with setup.scope():
        await _call(setup.agent("one"), "hello", thread)
        first = setup.recorded()["arguments"]
        await _call(setup.agent("two"), "hello", thread)  # the same thread
        second = setup.recorded()["arguments"]
        assert first[first.index("--session-id") + 1] != second[second.index("--session-id") + 1]

        # a new instance of the same agent continues its session, as after a restart
        await _call(setup.agent("one"), "again", thread)
        third = setup.recorded()["arguments"]
        assert "--session-id" not in third
        assert third[third.index("--resume") + 1] == first[first.index("--session-id") + 1]


def test_agents_made_of_a_name_have_a_stable_identity() -> None:
    assert claude_code_agent("a b/c").identity.uri == claude_code_agent("a b/c").identity.uri
    assert claude_code_agent("a b/c").identity.uri == "agent://claude-code/a%20b%2Fc"
    assert claude_code_agent("one").identity.uri != claude_code_agent("two").identity.uri


@pytest.mark.asyncio
async def test_a_session_which_did_not_finish_is_resumed_not_started_again(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    setup = _Setup(
        tmp_path,
        scenario="hang",
        sessions=ClaudeCodeSessions(state_directory=str(state)),
    )
    agent = setup.agent("coder")
    thread = uuid4()

    async with setup.scope():
        stream = agent.call(input="hello", thread=thread)
        try:
            async for chunk in stream:
                if isinstance(chunk, TextContent):
                    break  # abandoned, the session exists but its run never reported a result

        finally:
            await stream.aclose()

    first = setup.recorded()["arguments"]
    assert "--session-id" in first

    setup.config = setup.config.updating(env={**setup.config.env, "FAKE_CLAUDE_SCENARIO": "echo"})
    async with setup.scope():
        await _call(agent, "again", thread)

    second = setup.recorded()["arguments"]
    assert "--session-id" not in second  # Claude Code refuses to start a session which exists
    assert second[second.index("--resume") + 1] == first[first.index("--session-id") + 1]
    assert second[second.index("--system-prompt-snapshot") + 1] == "off"


@pytest.mark.asyncio
async def test_a_session_is_started_until_claude_code_stored_it(tmp_path: Path) -> None:
    state = tmp_path / "state"
    setup = _Setup(
        tmp_path,
        scenario="noresult",
        sessions=ClaudeCodeSessions(state_directory=str(state)),
    )
    agent = setup.agent("coder")
    thread = uuid4()

    async with setup.scope():
        with pytest.raises(ClaudeCodeFailed):
            await _call(agent, "hello", thread)

        for transcript in (state / "claude").rglob("*.jsonl"):
            transcript.unlink()  # nothing was stored by the run

        with pytest.raises(ClaudeCodeFailed):
            await _call(agent, "again", thread)

    assert "--session-id" in setup.recorded()["arguments"]
    assert "--resume" not in setup.recorded()["arguments"]


@pytest.mark.asyncio
async def test_the_text_of_the_result_is_streamed_when_nothing_else_was(tmp_path: Path) -> None:
    async def running(launch: Any) -> Any:
        yield ClaudeCodeResult(
            session_id="s",
            outcome=ClaudeCodeSuccess(text="only the result says it"),
            turns=1,
        )

    memory = AgentMemory.volatile()
    agent = claude_code_agent("coder", memory=memory)
    thread = uuid4()

    async with ctx.scope(
        "test",
        ClaudeCodeConfig(),
        ClaudeCodeRunner(binary="/fake", running=running),
    ):
        chunks = await _call(agent, "hello", thread)
        from draive.agents import AgentThread

        recalled = await memory.recall(
            thread=AgentThread.of(thread, agent_uri=agent.identity.uri),
            context=(),
        )

    assert _texts(chunks) == ["only the result says it"]
    assert recalled[-1].content.to_str() == "only the result says it"


@pytest.mark.asyncio
async def test_a_relative_state_directory_is_resolved_for_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    setup = _Setup(tmp_path, sessions=ClaudeCodeSessions(state_directory="relative-state"))

    async with setup.scope():
        await _call(setup.agent("coder"))

    recorded = setup.recorded()
    expected = (tmp_path / "work" / "relative-state").resolve()
    assert recorded["directory"] == str(expected / "workspace")
    assert recorded["environment"]["HOME"] == str(expected / "home")
    assert recorded["environment"]["CLAUDE_CONFIG_DIR"] == str(expected / "claude")


@pytest.mark.asyncio
async def test_stored_sessions_are_found_by_their_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "projects" / "some-project" / "abc-123.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}")

    assert await session_exists(str(tmp_path), "abc-123")
    assert not await session_exists(str(tmp_path), "other")
    assert not await session_exists(str(tmp_path / "missing"), "abc-123")
