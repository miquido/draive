# Installation

## Requirements

Draive requires Python 3.12 or higher. The framework is built on top of [Haiway](https://github.com/miquido/haiway) for state management and structured concurrency.

## Basic Installation

Install Draive using pip:

```bash
pip install draive
```

## Optional Dependencies

Draive provides integrations with various AI service providers and tools. Install only what you need:

### AI Service Providers

#### OpenAI
For GPT models, DALL-E, and embeddings (includes Azure OpenAI support):
```bash
pip install 'draive[openai]'
```

For OpenAI Realtime API support:
```bash
pip install 'draive[openai_realtime]'
```

#### Anthropic
For Claude models:
```bash
pip install 'draive[anthropic]'
```

For Claude via AWS Bedrock:
```bash
pip install 'draive[anthropic_bedrock]'
```

#### Google Gemini
For Gemini models via Google AI Studio:
```bash
pip install 'draive[gemini]'
```

#### Mistral
For Mistral models (includes Azure support):
```bash
pip install 'draive[mistral]'
```

#### Cohere
For Cohere models:
```bash
pip install 'draive[cohere]'
```

For Cohere via AWS Bedrock:
```bash
pip install 'draive[cohere_bedrock]'
```

#### Ollama
For local models via Ollama:
```bash
pip install 'draive[ollama]'
```

#### VLLM
For VLLM inference server (uses OpenAI-compatible API):
```bash
pip install 'draive[vllm]'
```

#### AWS Bedrock
For AWS Bedrock service (base integration):
```bash
pip install 'draive[bedrock]'
```

### Additional Features

#### Model Context Protocol (MCP)
For MCP client and server support:
```bash
pip install 'draive[mcp]'
```

#### OpenTelemetry
For observability and monitoring:
```bash
pip install 'draive[opentelemetry]'
```

## Installation Combinations

You can install multiple integrations together:

```bash
# For multi-provider support
pip install 'draive[openai,anthropic,gemini]'

# For AWS-based services
pip install 'draive[anthropic_bedrock,cohere_bedrock,bedrock]'
```

## Next Steps

After installation, proceed to the [Quickstart Guide](quickstart.md) to build your first Draive application.
