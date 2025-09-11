# Quickstart


```python
import asyncio
from draive import ctx, TextGeneration
from draive.openai import OpenAI, OpenAIResponsesConfig

async def main():
    # Create a context with OpenAI configuration
    async with ctx.scope(
        "quickstart",  # Context name
        OpenAIResponsesConfig(model="gpt-4o-mini"),  # Model configuration
        disposables=(OpenAI(),),  # Initialize OpenAI client
    ):
        # Generate text using the LLM
        result = await TextGeneration.generate(
            instructions="You are a helpful assistant",
            input="What is the capital of France?",
        )

        print(result)
        # Output: The capital of France is Paris.

if __name__ == "__main__":


```
# .env file
OPENAI_API_KEY=your-api-key-here
```

```python
from draive import load_env

# Load environment variables
load_env()

# Now you can use the client without hardcoding keys
async with ctx.scope("app", disposables=(OpenAI(),)):
    # Your code here


```python
from draive import tool
from datetime import datetime

@tool  # Decorator that converts function to LLM tool
async def get_current_time(timezone: str = "UTC") -> str:
    return f"Current time in {timezone}: {datetime.now().isoformat()}"

@tool
async def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

async def assistant_with_tools():
    async with ctx.scope(
        "assistant",
        disposables=(OpenAI(),),
        OpenAIResponsesConfig(model="gpt-4o-mini"),
    ):
        result = await TextGeneration.generate(
            instructions="You are a helpful assistant with access to tools",
            input="What time is it now? Also, what is 25 * 4?",
            tools=[get_current_time, calculate],  # Provide tools
        )

        print(result)
        # Output: The current time in UTC is 2024-03-15T10:30:00.


```python
from collections.abc import Sequence
from draive import ModelGeneration, DataModel

class PersonInfo(DataModel):
    name: str
    age: int
    occupation: str
    skills: Sequence[str]

async def extract_person_info():
    async with ctx.scope(
        "extraction",
        OpenAIResponsesConfig(model="gpt-4o-mini"),
        disposables=(OpenAI(),),
    ):
        person = await ModelGeneration.generate(
            PersonInfo,
            instructions="Extract person information from the text",
            input="""
            John Smith is a 32-year-old software engineer living in Seattle.
            He specializes in Python, machine learning, and cloud architecture.
            """,
        )

        print(f"Name: {person.name}")
        print(f"Age: {person.age}")
        print(f"Occupation: {person.occupation}")


```python
from draive.anthropic import Anthropic, AnthropicConfig
from draive.gemini import Gemini, GeminiConfig

# Using Anthropic Claude
async with ctx.scope(
    "claude-app",
    AnthropicConfig(model="claude-3-5-haiku-20241022"),
    disposables=(Anthropic(),),
):
    result = await TextGeneration.generate(
        instructions="You are Claude, a helpful AI assistant",
        input="Tell me a short joke",
    )
    print("Claude:", result)

# Using Google Gemini
async with ctx.scope(
    "gemini-app",
    GeminiConfig(model="gemini-2.5-flash"),
    disposables=(Gemini(),),
):
    result = await TextGeneration.generate(
        instructions="You are Gemini, a helpful AI assistant",
        input="Tell me a short joke",
    )


```python
from draive import Conversation, ConversationMessage

async def chat_example():
    async with ctx.scope(
        "chat",
        OpenAIResponsesConfig(model="gpt-4o-mini"),
        disposables=(OpenAI(),),
    ):
        response = await Conversation.completion(
            instructions="You are a helpful math tutor",
            input="What is calculus?",
            memory=[
                ConversationMessage.user("Hi!"),
                ConversationMessage.model("Hello, how can I help you?"),
            ]
        )

2. **Use Type Hints**: Draive leverages Python's type system for better IDE support
3. **Monitor Usage**: Use observability features to track token usage and costs

Ready to build something amazing? Dive deeper into Draive's features in the [First Steps](first-steps.md) guide!
