# Draive

[![PyPI](https://img.shields.io/pypi/v/draive)](https://pypi.org/project/draive/)
![Python Version](https://img.shields.io/badge/Python-3.12+-blue)
[![License](https://img.shields.io/github/license/miquido/draive)](https://github.com/miquido/draive/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/miquido/draive?style=social)](https://github.com/miquido/draive)

> **🏎️ Build production-grade LLM workflows and AI applications with confidence.**

Draive is a batteries-included Python framework for composing multi-model AI systems. It combines structured state management, typed parameters, multimodal content handling, and comprehensive observability so that teams can ship reliable AI features faster.

## Why teams choose Draive

- Unified abstraction layer across cloud, on-prem, and local LLM providers
- Immutable, strongly typed state powered by [Haiway](https://github.com/miquido/haiway)
- Guardrails, tool orchestration, RAG, and conversation management out of the box
- Seamless logging, metrics, and tracing via `ctx.log_*` and `ctx.record`
- Composable building blocks that scale from prototypes to mission-critical services

## Key capabilities

- **Multimodal content**: Work with text, images, audio, documents, and artifacts in a single, typed API.
- **Conversation flows**: Maintain contextual, stateful chat sessions tuned for completion and realtime use cases.
- **Retrieval & memory**: Include vector search, resource repositories, and memory utilities for RAG pipelines.
- **Tooling & orchestration**: Model execution, tool calling, retries, and error handling patterns that keep agents resilient.
- **Safety & governance**: Moderation, privacy, and quality guardrails configurable per workflow.
- **Operational excellence**: First-class observability, metrics, and structured concurrency to run AI in production.

## Provider coverage

Draive ships adapters with a shared interface for:

- OpenAI, Anthropic, Mistral (including Azure), Cohere (including Bedrock)
- Google Gemini (AI Studio), AWS Bedrock models, vLLM, Ollama for local models
- Bring-your-own providers by implementing the `GenerativeModel` protocol

## Architecture essentials

- Built on Haiway state containers (`State`, `ctx.scope`) for dependency injection without globals.
- Modules such as `draive/generation`, `draive/conversation`, `draive/embedding`, and `draive/guardrails` keep concerns separated and discoverable.
- Provider adapters live under `draive/<provider>/` and expose consistent config, client, and API layers.
- Utilities, resources, and multimodal helpers are ready to mix into your own stages and agents.

## Quick start

### Installation

```bash
pip install draive
```

Using [uv](https://github.com/astral-sh/uv)? Create an isolated environment and install Draive in one step:

```bash
uv venv
uv pip install draive
```

### Minimal example

```python
import asyncio
from draive import TextGeneration, ctx
from draive.openai import OpenAI, OpenAIResponsesConfig


async def main() -> None:
    async with ctx.scope(
        "quickstart",
        OpenAIResponsesConfig(model="gpt-4o-mini"),
        disposables=(OpenAI(),),
    ):
        response = await TextGeneration.generate(
            instructions="You are a branding assistant",
            input="Give me three tagline ideas for an AI travel app",
        )
        ctx.log_info("generated.response", content=response)


if __name__ == "__main__":
    asyncio.run(main())
```

Run the script with your preferred OpenAI credentials set via environment variables. Swap the configuration to another provider (e.g., Cohere, Gemini) without changing the rest of the code.

### Add retrieval or tools

- Use `draive.embedding` with `VectorIndex` to ingest documents and power RAG pipelines.
- Wire external actions with `draive.models.tools` and invoke them from `TextGeneration` or custom stages.
- Combine guardrails from `draive.guardrails` to validate outputs before returning them to end users.

## Where to go next

- Explore the [examples](./examples.md) for end-to-end agent, RAG, and multimodal templates.
- Dive into architecture guides under `docs/` to understand stages, state, and context patterns.
- Run `make format && make lint && make test` to verify your contributions before submitting a PR.
- Extend the framework by creating a new provider adapter or resource backend.

## Community & support

- File issues and track roadmap items on [GitHub](https://github.com/miquido/draive/issues).
- Join discussions, propose enhancements, or share your integrations via pull requests.
- Follow [Miquido](https://miquido.com) for updates and case studies powered by Draive.

**Built by [Miquido](https://miquido.com)** — empowering developers to build amazing AI applications.
