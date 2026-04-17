# 🏎️ draive 🏁

[![PyPI](https://img.shields.io/pypi/v/draive)](https://pypi.org/project/draive/)
![Python Version](https://img.shields.io/badge/Python-3.14+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
[![Docs](https://img.shields.io/badge/Documentation-yellow)](https://miquido.github.io/draive/)
![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/miquido/draive?utm_source=oss&utm_medium=github&utm_campaign=miquido%2Fdraive&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

🏎️ Build production-ready GenAI systems with strict typing, clean architecture, and fast iteration.

Draive helps teams move from prompt experiments to dependable AI features:
typed outputs, tool orchestration, multimodal workflows, evaluation, guardrails, and observability
in one cohesive framework.

If you want AI code that stays maintainable as your product grows, Draive is built for that.

## ✨ Why teams pick Draive

- **Typed by default**: generate validated `State` objects, not fragile string blobs.
- **Reliable execution model**: `ctx.scope(...)` gives explicit state, dependency lifecycle, and
    structured concurrency.
- **Model portability**: swap providers without rewriting your business workflow.
- **Real-world coverage**: tools, multimodal content, retrieval, evaluators, and guardrails are
    first-class building blocks.
- **Production visibility**: built-in hooks for logs, metrics, traces, and quality checks.

## 🚀 Quick start

Here’s what a simple Draive setup looks like:

```python
from draive import ctx, TextGeneration, tool
from draive.openai import OpenAI, OpenAIResponsesConfig

@tool # simply annotate a function as a tool
async def current_time(location: str) -> str:
    return f"Time in {location} is 9:53:22"

async with  ctx.scope( # create execution context
    "example", # give it a name
    OpenAIResponsesConfig(model="gpt-4o-mini"), # prepare configuration
    disposables=(OpenAI(),), # define resources and service clients available
):
    result: str = await TextGeneration.generate( # choose a right generation abstraction
        instructions="You are a helpful assistant", # provide clear instructions
        input="What is the time in Kraków?", # give it some input (including multimodal)
        tools=[current_time], # and select any tools you like
    )

    print(result) # to finally get the result!
    # output: The current time in Kraków is 9:53:22.
```

Read the [Draive Documentation](https://miquido.github.io/draive/) for all required knowledge.

For full examples, head over to the [Draive Examples](https://github.com/miquido/draive-examples)
repository.

## ❓ What is Draive good for

Draive is built for developers who want clarity, flexibility, and control when working with LLMs.

Whether you’re building an autonomous agent, automating data flow, extracting information from
documents, handling audio or images — Draive has your back.

### What you can build with Draive

- **🔁 Start with evaluation** - make sure your app behaves as expected, right from the start
- **🛠 Turn any Python function into a tool** that LLMs can call
- **🔄 Switch between providers** like OpenAI, Claude, Gemini, or Mistral in seconds
- **🧱 Design structured workflows** with reusable `Step` and `StepState` pipelines
- **🛡 Enforce output quality using guardrails** including moderation and runtime evaluation
- **📊 Monitor with ease** - plug into any OpenTelemetry-compatible services
- **⚙️ Control your context** - use on-the-fly LLM context modifications for best results

### What makes Draive stand out

- **Instruction Optimization**: Draive gives you clean ways to write and refine prompts, including
    metaprompts, instruction helpers, and optimizers. You can go from raw prompt text to reusable,
    structured config in no time.
- **Composable Workflows**: Build modular flows using `Step`, tools, and context-scoped state.
    Every piece is reusable, testable, and fits together cleanly.
- **Tooling = Just Python**: Define a tool by writing a function. Annotate it. That’s it. Draive
    handles the rest — serialization, context, and integration with LLMs.
- **Structured Outputs** - decode to typed Python `State` models with schema-aware generation.
- **Multimodal + Resource-Native**: Work with text, images, audio, files, and artifacts through one
    content model.
- **RAG Ready**: Built-in embeddings and `VectorIndex` utilities support retrieval-heavy workflows.
- **Telemetry + Evaluators**: Measure timing, quality, tool usage, and regressions as part of CI.
- **Model-Agnostic by Design**: Built-in support for major hosted and self-hosted providers.

## 🖥️ Install

With pip:

```bash
pip install draive
```

### Optional dependencies

Draive library comes with optional integrations to 3rd party services:

- OpenAI:

Use OpenAI services client, including GPT, dall-e and embedding. Allows to use Azure services as
well.

```bash
pip install 'draive[openai]'
```

- Anthropic:

Use Anthropic services client, including Claude.

```bash
pip install 'draive[anthropic]'
```

- Gemini:

Use Google AIStudio services client, including Gemini.

```bash
pip install 'draive[gemini]'
```

- Mistral:

Use Mistral services client. Allows to use Azure services as well.

```bash
pip install 'draive[mistral]'
```

- Cohere:

Use Cohere services client.

```bash
pip install 'draive[cohere]'
```

- Ollama:

Use Ollama services client.

```bash
pip install 'draive[ollama]'
```

- VLLM:

Use VLLM services through OpenAI client.

```bash
pip install 'draive[vllm]'
```

- Bedrock:

Use AWS Bedrock-backed models.

```bash
pip install 'draive[bedrock]'
```

- MCP:

Use Model Context Protocol integrations.

```bash
pip install 'draive[mcp]'
```

- OpenTelemetry:

Export traces/metrics to your observability stack.

```bash
pip install 'draive[opentelemetry]'
```

- Postgres:

Use Postgres-backed persistence helpers.

```bash
pip install 'draive[postgres]'
```

- Qdrant:

Use Qdrant vector database integration.

```bash
pip install 'draive[qdrant]'
```

## 👷 Contributing

Draive is open-source and always growing — and we’d love your help.

Got an idea for a new feature? Spotted a bug? Want to improve the docs or share an example? Awesome.
Open a PR or start a discussion — no contribution is too small!

Whether you're fixing typos, building new integrations, or just testing things out and giving
feedback — you're welcome here.

## Community & Support

- Report issues and discuss ideas on [GitHub](https://github.com/miquido/draive/issues).
- Learn more about the team behind Draive at [Miquido](https://miquido.com).

**Built by [Miquido](https://miquido.com)**

## License

MIT License

Copyright (c) 2024-2025 Miquido

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
