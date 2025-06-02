# 🏎️ draive 🏁

![Python Version](https://img.shields.io/badge/Python-3.12+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/miquido/draive?utm_source=oss&utm_medium=github&utm_campaign=miquido%2Fdraive&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

🏎️ A fast, flexible Python library for building powerful LLM workflows and AI apps. 🏎️

Draive gives you everything you need to turn large language models into production-ready software: structured workflows, tool use, instruction refinement, guardrails, observability, and seamless multi-model integration — all in one clean, composable package.

If you've ever felt like you’re stitching together glue code and hoping for the best — Draive is what you wish you'd started with!

## 🚀 Quick start

Dive straight into the code and learn how to use draive with our interactive [guides](https://github.com/miquido/draive/tree/main/guides).
For quick solutions to common problems, explore our [cookbooks](https://github.com/miquido/draive/tree/main/cookbooks).

Here’s what a simple Draive setup looks like:

```python
from draive import ctx, TextGeneration, tool
from draive.openai import OpenAI, OpenAIChatConfig


@tool # simply annotate a function as a tool
async def current_time(location: str) -> str:
    return f"Time in {location} is 9:53:22"

async with  ctx.scope( # create execution context
    "example", # give it a name
    OpenAIChatConfig(model="gpt-4o-mini"), # prepare configuration
    disposables=[OpenAI()], # define resources and service clients available
):
    result: str = await TextGeneration.generate( # choose a right generation abstraction
        instruction="You are a helpful assistant", # provide clear instructions
        input="What is the time in Kraków?", # give it some input (including multimodal)
        tools=[current_time], # and select any tools you like
    )

    print(result) # to finally get the result!
    # output: The current time in Kraków is 9:53:22.
```

For full examples, head over to the [Draive Examples](https://github.com/miquido/draive-examples) repository.

## ❓ What is Draive good for?

Draive is built for developers who want clarity, flexibility, and control when working with LLMs.

Whether you’re building an autonomous agent, automating customer support, generating structured data, or routing across multiple models — Draive has your back.

### What you can do with Draive:

- **🔁 Write smarter instructions** — and refine them programmatically
- **🛠 Turn any Python function into a tool** that LLMs can call
- **🔄 Switch between providers** like OpenAI, Claude, Gemini, or Mistral in seconds
- **🧱 Design structured workflows** using stages, retries, and routing logic
- **🛡 Enforce output structure using guardrails** (e.g., typed JSON)
- **📊 Monitor everything with telemetry** and evaluation tools
- **⚙️ Make your flows configuration-driven** and easy to test or deploy

### Why you'll like it:

- **Instruction Optimization**: Draive gives you clean ways to write and refine prompts, including metaprompts, instruction helpers, and optimizers. You can go from raw prompt text to a reusable, structured config in no time.
- **Composable Workflows**: Use Stage, Router, and Tool components to build modular flows. Each piece is reusable, debuggable, and works with the rest of your Python stack.
- **Tooling = Just Python**: Define a tool by writing a function. Annotate it. That’s it. Draive handles the rest — serialization, context, and integration with LLMs.
- **Telemetry + Evaluators**: Draive logs everything you care about: timing, output shape, tool usage, error cases. Evaluators let you benchmark or regression-test LLM behavior like a normal part of your CI.
- **Model-Agnostic by Design**: Built-in support for most major providers.

## 🧱 What can you build with draive?

These are the kinds of projects Draive was built for:

- **AI agent for content marketing**: Automates blog post drafts, inserts SEO keywords, fetches trends via tools, and publishes to a CMS.
- **Developer assistant bot**: Parses pull requests, summarizes diffs, generates release notes, and responds to GitHub issues.
- **AI-powered customer support**: Reads company docs, routes queries, and calls real tools (like product status or order lookup) to respond accurately.
- **Document parser for finance teams**: Converts PDFs to structured JSON using multimodal input, validates format, and sends summaries via Slack.
- **Realtime dashboard builder**: Accepts natural language queries and builds charts from your own data sources using custom tools.
- and much, much more!

## 🖥️ Install

With pip:

```bash
pip install draive
```

### Optional dependencies

Draive library comes with optional integrations to 3rd party services:

- OpenAI:

Use OpenAI services client, including GPT, dall-e and embedding. Allows to use Azure services as well.

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

## 👷 Contributing

Draive is open-source and always growing — and we’d love your help.

Got an idea for a new feature? Spotted a bug? Want to improve the docs or share an example? Awesome. Open a PR or start a discussion — no contribution is too small!

Whether you're fixing typos, building new integrations, or just testing things out and giving feedback — you're welcome here.


## License

MIT License

Copyright (c) 2024-2025 Miquido

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
