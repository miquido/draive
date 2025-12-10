# Installation

Install Draive into a dedicated Python 3.13+ environment so your project dependencies remain
isolated. We recommend [uv](https://github.com/astral-sh/uv) for deterministic installs, but
standard `pip` works as well.

## Using pip

```bash
python -m pip install --upgrade pip
pip install "draive"
```

Add provider extras to pull in the SDKs you need:

```bash
pip install "draive[openai]"
pip install "draive[anthropic]"
pip install "draive[gemini]"
```

Combine extras for multi-provider projects:

```bash
pip install "draive[openai,anthropic,gemini]"
```

## Using uv

```bash
uv venv
uv pip install "draive[openai,anthropic,gemini]"
```

To sync an existing project with the pinned dependencies from `pyproject.toml`:

```bash
uv sync --all-groups --all-extras --frozen
```

## Optional extras

- `draive[openai]`, `draive[openai_realtime]` for OpenAI Responses/Realtime.
- `draive[anthropic]`, `draive[anthropic_bedrock]` for Claude models (direct or via Bedrock).
- `draive[mistral]`, `draive[gemini]`, `draive[cohere]`, `draive[cohere_bedrock]` for other hosted
    LLMs.
- `draive[bedrock]`, `draive[aws]` for AWS model/runtime integrations.
- `draive[ollama]`, `draive[vllm]` for local or self-hosted deployments.
- `draive[qdrant]`, `draive[postgres]` for vector/storage backends; add `pgvector` separately where
    needed.
- `draive[httpx]`, `draive[mcp]`, `draive[opentelemetry]`, `draive[docs]` for HTTP utilities, MCP,
    tracing, and docs site builds.

## Verify your environment

```bash
python -c "import draive; print(draive.__version__)"
```

You are ready to jump into the [quickstart](./quickstart.md) once the version prints without errors.
