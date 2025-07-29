# Draive

[![PyPI](https://img.shields.io/pypi/v/draive)](https://pypi.org/project/draive/)
![Python Version](https://img.shields.io/badge/Python-3.12+-blue)
[![License](https://img.shields.io/github/license/miquido/draive)](https://github.com/miquido/draive/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/miquido/draive?style=social)](https://github.com/miquido/draive)

**🏎️ All-in-one, flexible Python library for building powerful LLM workflows and AI apps 🏎️**

Draive gives you everything you need to turn large language models into production-ready software: agents, structured workflows, tool use, instruction refinement, guardrails, observability, and seamless multi-model integration — all in one clean, composable package.

If you've ever felt like you're stitching together glue code and hoping for the best — Draive is what you wish you'd started with!

## Why Choose Draive?

Building LLM applications often involves complex prompt management, model switching, tool integration, and quality control. Draive simplifies these challenges through:

### 🧱 **Composable by Design**
Build modular flows using reusable components. Every piece is testable and fits together seamlessly.

### 🔄 **Multi-Model Ready**
Switch between OpenAI, Claude, Gemini, Mistral, and more in seconds. No vendor lock-in.

### 🛠️ **Tools = Just Python**
Define a tool by writing a function. Annotate it. Done. Draive handles serialization, context, and LLM integration.

### 🛡️ **Production Ready**
Built-in evaluation, guardrails, observability, and error handling for reliable applications.

### 🎯 **Developer Experience**
Full type safety, excellent IDE support, and clear error messages. Built on solid Haiway foundations.

## Core Features

### 🤖 **Multi-Provider LLM Support**
- OpenAI (GPT, DALL-E, Embeddings, Realtime API)
- Anthropic (Claude, including Bedrock)
- Google (Gemini via AI Studio)
- Mistral (including Azure)
- Cohere (including Bedrock)
- Ollama (local models)
- VLLM (inference servers)
- AWS Bedrock (multiple providers)

### 🔧 **Advanced Capabilities**
- **Tools & Function Calling**: Turn any Python function into an LLM tool
- **Structured Generation**: Generate validated Python objects from LLMs
- **Multimodal Content**: Handle text, images, audio, and documents
- **Conversation Management**: Build stateful, context-aware chat applications
- **RAG & Vector Search**: Built-in document indexing and retrieval
- **Evaluation Framework**: Test and improve LLM outputs systematically
- **MCP Integration**: Connect to the Model Context Protocol ecosystem

### 🎭 **Production Features**
- **Guardrails**: Content moderation, privacy protection, quality control
- **Observability**: OpenTelemetry integration and detailed metrics
- **Retry & Error Handling**: Robust failure recovery patterns
- **Streaming**: Real-time response generation
- **Context Management**: Immutable state and dependency injection

## Getting Started

### 📥 [Installation](getting-started/installation.md)
```bash
pip install 'draive[openai]'  # or anthropic, gemini, etc.
```
Set up Draive with the integrations you need

### 🚀 [Quickstart](getting-started/quickstart.md)
Build your first AI assistant in minutes with our hands-on tutorial

### 📚 [First Steps](getting-started/first-steps.md)
Master core concepts with practical examples and patterns

## What You Can Build

### 🤖 **Autonomous Agents**
Create agents that can reason, plan, and execute complex tasks using tools and structured workflows.

### 💬 **Conversational AI**
Build chatbots, customer support systems, and interactive assistants with memory and context.

### 📊 **Data Processing**
Extract structured information from documents, analyze content, and transform unstructured data.

### 🔍 **RAG Applications**
Create knowledge bases that can answer questions from your documents and data sources.

### 🛡️ **Quality Control**
Implement evaluation pipelines to ensure your LLM applications meet quality standards.

## Learn Draive

### Essential Guides
- **[Basic Usage](guides/BasicUsage.md)** - Your first text generation and model usage
- **[Tools & Functions](guides/BasicToolsUse.md)** - Turn Python functions into LLM tools
- **[Conversations](guides/BasicConversation.md)** - Build interactive chat applications
- **[State Management](guides/AdvancedState.md)** - Immutable data and context systems

### Practical Cookbooks
- **[RAG System](cookbooks/BasicRAG.md)** - Build a document search and retrieval system
- **[MCP Integration](cookbooks/BasicMCP.md)** - Connect to external tool ecosystems
- **[Data Extraction](cookbooks/BasicDataExtraction.md)** - Extract structured data from text
- **[Evaluation](guides/BasicEvaluation.md)** - Test and improve your applications

## When to Use Draive

✅ **Perfect for:**
- LLM-powered applications and services
- Multi-model AI workflows
- Applications requiring tool integration
- RAG and knowledge management systems
- AI agents and autonomous workflows
- Content generation and analysis
- Conversational AI and chatbots

⚠️ **Consider alternatives for:**
- Simple, one-off LLM API calls
- Applications not using language models

## Architecture

Draive is built on **[Haiway](https://github.com/miquido/haiway)**, providing:

- **Immutable State Management**: Type-safe, validated data structures
- **Context-based Dependency Injection**: Clean state propagation
- **Structured Concurrency**: Automatic resource management
- **Functional Approach**: Pure functions over stateful objects

## Community & Support

### 💻 [GitHub Repository](https://github.com/miquido/draive)
Source code, issues, and contributions

### 🎯 [Examples Repository](https://github.com/miquido/draive-examples)
Complete applications and use case examples

## Contributing

Draive is open-source and growing! Whether you're fixing bugs, adding features, improving docs, or sharing examples — all contributions are welcome.

**Built by [Miquido](https://miquido.com)**
Empowering developers to build amazing AI applications
