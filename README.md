# 🏎️ draive 🏁

🏎️ Fast-track your LLM-based apps with an accessible, production-ready library. 🏎️

Are you looking for maximum flexibility and efficiency in your next Python library? Tired of unnecessary complexities and inefficient token usage?

👉 Introducing **draive** - an open-source Python library under the Miquido AI Kickstarter framework, designed to simplify and accelerate the development of LLM-based applications. Get started with draive to streamline your workflow and build powerful, efficient apps with ease.

## 🚀 Quick start

Dive straight into the code and learn how to use draive with our interactive [guides](https://github.com/miquido/draive/tree/main/guides). For quick solutions to common problems, explore our [cookbooks](https://github.com/miquido/draive/tree/main/cookbooks).

Great, but how it looks like?

```python
from draive import ctx, generate_text, LMM, tool, Toolbox
from draive.openai import OpenAIClient, openai_lmm_invocation


@tool # simply annotate a function as a tool
async def current_time(location: str) -> str:
    return f"Time in {location} is 9:53:22"

async with ctx.new( # create execution context
    dependencies=[OpenAIClient], # define client for llm provider
    state=[LMM(invocation=openai_lmm_invocation)], # and use it for this scope
):
    result: str = await generate_text( # choose the right abstraction, like simple `generate_text`
        instruction="You are a helpful assistant", # provide clear instructions
        input="What is the time in Kraków?", # give it some input (including multimodal)
        tools=Toolbox(current_time), # and select any tools you like
    )

    print(result) # to finally get the result!
    # output: The current time in Kraków is 9:53:22.
```

Fully functional examples of using the Draive library are also available in [Draive Examples](https://github.com/miquido/draive-examples) repository.

## ❓ What is draive?

**draive** is an open-source Python library for developing apps powered by large language models. It stands out for its simplicity, consistent behavior, and transparency.

### Key Features:

- **🧱 Abstract building blocks**: Easily connect multiple functionalities with LLMs and link various LLMs together.
- **🧩 Flexible integration**: Supports any LLM, external service, and other AI solutions.
- **🧒 User-friendly framework**: Designed to build scalable and composable data processing pipelines with ease.
- **⚙️ Function-oriented design**: Utilizes basic programming concepts, allowing you to represent complex programs as simple functions.
- **🏗️ Composable and reusable**: Combine functions to create complex programs, while retaining the ability to use them individually.
- **📊 Diagnostics and metrics**: Offers extensive tools for measuring and debugging complex functionalities.
- **🔄 Fully typed and asynchronous**: Ensures type safety and efficient asynchronous operations for modern Python apps.

## 🧱 What can you build with draive?

### 🦾 RAG applications

RAG enhances model capabilities and personalizes the outputs.

- **Examples**: Question answering, custom knowledge bases.

### 🧹 Extracting structured output

Simplified data extraction and structuring.

- **Examples**: Data parsing, report generation.

### 🤖 Chatbots

Sophisticated conversational agents.

- **Examples**: Customer service bots, virtual assistants.

… and much more!

## 🖥️ Install

With pip:

```bash
pip install draive
```

### Optional dependencies

- OpenAI:

```bash
pip install draive[openai]
```

- Anthropic:

```bash
pip install draive[anthropic]
```

- Gemini:

```bash
pip install draive[gemini]
```

- Mistral:

```bash
pip install draive[mistral]
```

- Ollama:

```bash
pip install draive[ollama]
```

- Mistral.rs:

```bash
pip install draive[mistralrs]
```

- Fastembed:

```bash
pip install draive[fastembed]
```

## 👷 Contributing

As an open-source project in a rapidly evolving field, we welcome all contributions. Whether you can add a new feature, enhance our infrastructure, or improve our documentation, your input is valuable to us.

We welcome any feedback and suggestions! Feel free to open an issue or pull request.

## License

MIT License

Copyright (c) 2024 Miquido

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
