#!/usr/bin/env python3
#
# /// script
# dependencies = [
#     "draive[openai,anthropic,gemini]",
# ]
# ///
#
"""
Draive Instruction Generator

A standalone script that uses draive's prompt generator to create detailed, actionable
instructions for LLMs from simple task descriptions.

SETUP:
------
This script requires environment variables for API access:

For Anthropic:
    export ANTHROPIC_API_KEY="your-api-key-here"

For OpenAI:
    export OPENAI_API_KEY="your-api-key-here"

For Gemini:
    export GOOGLE_API_KEY="your-api-key-here"

You can also provide local .env file next to this script, it will load variables automatically.

USAGE:
------
# From command line argument:
uv run generate_instruction.py --provider Anthropic "Create a function to validate email addresses"

# From stdin (useful for piping):
echo "Write unit tests for authentication" | uv run generate_instruction.py --provider OpenAI

# With custom model:
uv run generate_instruction.py --provider Anthropic --model claude-sonnet-4-20250514 "Custom task"

SUPPORTED PROVIDERS:
-------------------
- Anthropic: Uses Claude models for instruction generation
- OpenAI: Uses GPT models for instruction generation
- Gemini: Uses Google's Gemini models for instruction generation

OUTPUT:
-------
The script outputs professionally formatted instructions that include:
- Clear task descriptions and requirements
- Step-by-step implementation guidance
- Code examples and templates
- Best practices and guidelines
- Error handling recommendations

The generated instructions are optimized for LLM consumption and can be used
directly as system prompts or task descriptions.
"""

import argparse
import asyncio
import sys

from draive import ctx, load_env, setup_logging
from draive.anthropic import Anthropic, AnthropicConfig
from draive.gemini import Gemini, GeminiConfig
from draive.helpers import prepare_instructions
from draive.openai import OpenAI, OpenAIResponsesConfig


async def generate_with_provider(
    provider: str,
    description: str,
    model: str | None = None,
) -> str:
    # Configure provider context
    if provider.lower() == "anthropic":
        model_name = model or "claude-sonnet-4-20250514"
        async with ctx.scope(
            "generate_instruction",
            AnthropicConfig(model=model_name),
            disposables=(Anthropic(),),
        ):
            instructions = await prepare_instructions(description)
            return instructions

    elif provider.lower() == "openai":
        model_name = model or "o3"
        async with ctx.scope(
            "generate_instruction",
            OpenAIResponsesConfig(model=model_name),
            disposables=(OpenAI(),),
        ):
            instructions = await prepare_instructions(description)
            return instructions

    elif provider.lower() == "gemini":
        model_name = model or "gemini-2.5-pro"
        async with ctx.scope(
            "generate_instruction",
            GeminiConfig(model=model_name),
            disposables=(Gemini(),),
        ):
            instructions = await prepare_instructions(description)
            return instructions

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def main():
    load_env()
    setup_logging("generate_instruction")

    parser = argparse.ArgumentParser(
        description="Generate instructions using draive's prompt generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run generate_instruction.py --provider Anthropic "Create a function to validate email addresses"
  echo "Write unit tests" | uv run generate_instruction.py --provider OpenAI
  uv run generate_instruction.py --provider Gemini "Explain machine learning concepts"
  uv run generate_instruction.py --provider OpenAI --model gpt-4o "Complex coding task"
        """.strip(),  # noqa: E501
    )

    parser.add_argument(
        "--provider",
        choices=["Anthropic", "OpenAI", "Gemini"],
        required=True,
        help="AI provider to use for instruction generation",
    )

    parser.add_argument(
        "--model",
        help="Specific model to use (overrides default model for provider)",
    )

    parser.add_argument(
        "description",
        nargs="?",
        help="Description of the instruction to generate (if not provided, reads from stdin)",
    )

    args = parser.parse_args()

    # Get description from args or stdin
    if args.description:
        description = args.description

    else:
        if sys.stdin.isatty():
            print("Error: No description provided and no input from stdin", file=sys.stderr)
            print("Use --help for usage information", file=sys.stderr)
            sys.exit(1)

        description = sys.stdin.read().strip()

    if not description:
        print("Error: Empty description provided", file=sys.stderr)
        sys.exit(1)

    try:
        # Generate instruction
        result = asyncio.run(generate_with_provider(args.provider, description, args.model))
        print(result)

    except KeyboardInterrupt:
        print("\nOperation cancelled", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
