# Installation

Install Draive into a dedicated Python 3.12+ environment so your project dependencies remain
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

- `draive[ollama]` for local Ollama deployments.
- `draive[cohere]` or `draive[cohere_bedrock]` for Cohere hosted or AWS Bedrock access.
- `draive[postgres]` when you need the PostgreSQL resource backend.
- `draive[docs]` to render documentation locally with MkDocs.

## Verify your environment

```bash
python -c "import draive; print(draive.__version__)"
```

You are ready to jump into the [quickstart](./quickstart.md) once the version prints without errors.
