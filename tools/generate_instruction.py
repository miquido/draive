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
uv run generate_instruction.py --provider openai "Create a function to validate email addresses"

# From stdin (useful for piping):
echo "Write unit tests for authentication" | uv run generate_instruction.py --provider openai

# With custom model:
uv run generate_instruction.py --provider anthropic --model claude-sonnet-4-20250514 "Custom task"

# Injecting guardrails/guidelines:
uv run generate_instruction.py --provider openai --guidelines "Prefer Python" "Port legacy pipeline"

SUPPORTED PROVIDERS:
-------------------
- Anthropic: Uses Claude 3.5 models for instruction generation (default: claude-sonnet-4-20250514)
- OpenAI: Uses GPT-5 models for instruction generation (default: gpt-5-mini)
- Gemini: Uses Gemini 2.x models for instruction generation (default: gemini-2.0-flash)

OUTPUT:
-------
The script outputs professionally formatted instructions that include:
- Clear task descriptions and requirements
- Step-by-step implementation guidance
- Code examples and templates
- Best practices and guidelines
- Error handling recommendations
- Follow-up questions when clarification is required

The generated instructions are optimized for LLM consumption and can be used
directly as system prompts or task descriptions.
"""

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass

from draive import Disposable, State, ctx, load_env, setup_logging
from draive.anthropic import Anthropic, AnthropicConfig
from draive.gemini import Gemini, GeminiConfig
from draive.helpers import InstructionPreparationAmbiguity, prepare_instructions
from draive.openai import OpenAI, OpenAIResponsesConfig


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    display_name: str
    default_model: str
    config_factory: Callable[[str], State]
    disposable_factory: Callable[[], Disposable]

    def create_config(self, model: str | None) -> State:
        return self.config_factory(model or self.default_model)

    def create_disposable(self) -> Disposable:
        return self.disposable_factory()


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        key="anthropic",
        display_name="Anthropic",
        default_model="claude-sonnet-4-20250514",
        config_factory=lambda model: AnthropicConfig(model=model),
        disposable_factory=Anthropic,
    ),
    "openai": ProviderSpec(
        key="openai",
        display_name="OpenAI",
        default_model="gpt-5-mini",
        config_factory=lambda model: OpenAIResponsesConfig(model=model),
        disposable_factory=OpenAI,
    ),
    "gemini": ProviderSpec(
        key="gemini",
        display_name="Gemini",
        default_model="gemini-2.0-flash",
        config_factory=lambda model: GeminiConfig(model=model),
        disposable_factory=Gemini,
    ),
}


def _parse_provider(value: str) -> ProviderSpec:
    key = value.strip().lower()
    if key not in PROVIDERS:
        available = ", ".join(spec.display_name for spec in PROVIDERS.values())
        raise argparse.ArgumentTypeError(
            f"Unsupported provider '{value}'. Available providers: {available}"
        )

    return PROVIDERS[key]


async def generate_with_provider(
    provider: ProviderSpec,
    description: str,
    *,
    model: str | None = None,
    guidelines: str | None = None,
) -> str:
    async with ctx.scope(
        "generate_instruction",
        provider.create_config(model),
        disposables=(provider.create_disposable(),),
    ):
        return await prepare_instructions(description, guidelines=guidelines)


def main():
    load_env()
    setup_logging("generate_instruction")

    parser = argparse.ArgumentParser(
        description="Generate instructions using draive's prompt generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run generate_instruction.py --provider openai "Create a function to validate email addresses"
  echo "Write unit tests" | uv run generate_instruction.py --provider openai
  uv run generate_instruction.py --provider gemini "Explain machine learning concepts"
  uv run generate_instruction.py --provider openai --model gpt-5 "Complex coding task"
  uv run generate_instruction.py --provider anthropic --guidelines "Prefer TypeScript" "Ship auth flow"
        """.strip(),  # noqa: E501
    )

    provider_choices = ", ".join(spec.display_name for spec in PROVIDERS.values())
    parser.add_argument(
        "--provider",
        type=_parse_provider,
        required=True,
        help=f"AI provider to use for instruction generation ({provider_choices})",
        metavar="PROVIDER",
    )

    parser.add_argument(
        "--model",
        help="Specific model to use (overrides default model for provider)",
    )

    parser.add_argument(
        "--guidelines",
        help="Optional guidelines that should be considered during instruction preparation",
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
            sys.stderr.write("Error: No description provided and no input from stdin\n")
            sys.stderr.write("Use --help for usage information\n")
            sys.exit(1)

        description = sys.stdin.read().strip()

    if not description:
        sys.stderr.write("Error: Empty description provided\n")
        sys.exit(1)

    try:
        # Generate instruction
        provider = args.provider
        model_name = args.model
        result = asyncio.run(
            generate_with_provider(
                provider,
                description,
                model=model_name,
                guidelines=args.guidelines,
            )
        )
        sys.stdout.write(result.rstrip("\n") + "\n")

    except KeyboardInterrupt:
        sys.stderr.write("\nOperation cancelled\n")
        sys.exit(130)

    except InstructionPreparationAmbiguity as exc:
        sys.stderr.write("Instruction requires clarification. Follow-up questions:\n")
        sys.stderr.write(exc.questions.rstrip("\n") + "\n")
        sys.exit(2)

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
