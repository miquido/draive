# Getting Started

Welcome to Draive, a Haiway-powered toolkit for building reliable, observable AI applications. This
guide gives you the context you need before diving into installation and hands-on tutorials.

## Why Draive

- **Composable foundations**: Build on immutable `State` objects and scoped context managers, so
    every component remains testable and predictable.
- **Provider flexibility**: Switch between OpenAI, Anthropic, Gemini, Mistral, local models, and
    more without rewriting your pipelines.
- **Multimodal-first**: Work with text, audio, images, and artifacts through a unified content API.
- **Production guardrails**: Apply moderation, privacy, validation, and telemetry with the same
    abstractions you use for generation.

## What You Will Build

Throughout the getting-started journey you will assemble:

- A basic assistant that can call your own Python tools.
- A multimodal workflow that combines text and image inputs.
- A retrieval-augmented generation (RAG) flow over your documents.
- A structured extraction pipeline that yields typed outputs you can store or downstream.
- A test and evaluation harness to keep regressions out of production.

## Prerequisites

- Python 3.13+ with `uv` managing the virtual environment at `./.venv` (this repository includes a
    ready-to-use setup).
- Familiarity with `async`/`await` and running `pytest` from the command line.
- Access tokens for the model providers you plan to use (store them in environment variables—never
    hard-code secrets).

## Core Concepts

1. **State management** – mutable-looking, immutable-under-the-hood `State` classes represent
    configuration and runtime data. You update them with methods like `State.updated(...)` to keep
    histories, snapshots, and metrics consistent.
1. **Context scoping** – `ctx.scope(...)` activates a stack of `State` instances and disposables for
    a logical unit of work, ensuring structured concurrency and clean teardown.
1. **Generation flows** – typed facades in `draive.generation` orchestrate text, image, and audio
    calls, while provider adapters translate the request to each backend.
1. **Tools and multimodal content** – `MultimodalContent`, `ResourceContent`, and tool abstractions
    let you stream artifacts, call Python functions, or chain agents without sacrificing type
    safety.
1. **Guardrails and observability** – moderation, privacy, metrics, and logging integrations keep
    your application auditable. Use `ctx.log_*` for structured logs and `ctx.record` for metrics.

## Next Steps

1. Follow the [Installation](installation.md) guide to set up dependencies and the runtime
    environment.
1. Walk through the quickstart notebooks and examples under `docs/cookbooks/` to see Draive in
    action.
1. Explore provider-specific instructions in `docs/guides/` when you are ready to connect to
    production endpoints.

You now have the core mental model for Draive. Continue with installation to bring the toolkit to
life.
