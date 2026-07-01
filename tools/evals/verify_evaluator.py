#!/usr/bin/env python3
#
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "draive[openai,anthropic,gemini,mistral,ollama]",
#     "pyyaml~=6.0",
# ]
# ///
#
"""
Human baseline verification entrypoint.

Runs a single draive LLM-based evaluator across a YAML baseline file and
prints a Cohen's kappa report comparing the evaluator's scores to the
human-labeled ground truth.

USAGE
-----
    uv run tools/evals/verify_evaluator.py \\
        --provider openai \\
        --baseline tools/evals/baselines/coherence.yaml

    uv run tools/evals/verify_evaluator.py \\
        --provider anthropic \\
        --model claude-sonnet-4-5 \\
        --baseline path/to/my_relevance_baseline.yaml \\
        --concurrency 8

    # List all evaluators that have a registered baseline mapping:
    uv run tools/evals/verify_evaluator.py --list

ENVIRONMENT
-----------
Set the API key for the chosen provider:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY
A local `.env` next to the script is loaded automatically.
"""

import argparse
import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

# Allow `python tools/evals/verify_evaluator.py ...` style invocation by making
# the `tools` directory importable as a top-level package.
_TOOLS_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from haiway import Disposable, State, load_env, setup_logging  # noqa: E402

from draive.anthropic import Anthropic, AnthropicConfig  # noqa: E402
from draive.gemini import Gemini, GeminiConfig  # noqa: E402
from draive.mistral import Mistral, MistralChatConfig  # noqa: E402
from draive.ollama import Ollama, OllamaChatConfig  # noqa: E402
from draive.openai import OpenAI, OpenAIResponsesConfig  # noqa: E402
from evals.baseline import BaselineDocument, load_baseline  # noqa: E402
from evals.registry import available_evaluators, lookup_evaluator  # noqa: E402
from evals.runner import SuiteResult, run_suite  # noqa: E402


class ProviderSpec(State):
    key: str
    display_name: str
    default_model: str
    config_factory: Callable[[str], State]
    disposable_factory: Callable[[], Disposable]

    def create_config(self, model: str | None, /) -> State:
        return self.config_factory(model or self.default_model)

    def create_disposable(self) -> Disposable:
        return self.disposable_factory()


PROVIDERS: Final[dict[str, ProviderSpec]] = {
    "anthropic": ProviderSpec(
        key="anthropic",
        display_name="Anthropic",
        default_model="claude-sonnet-4-5",
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
        default_model="gemini-3.5-flash",
        config_factory=lambda model: GeminiConfig(model=model),
        disposable_factory=Gemini,
    ),
    "mistral": ProviderSpec(
        key="mistral",
        display_name="Mistral",
        default_model="mistral-small-latest",
        config_factory=lambda model: MistralChatConfig(model=model),
        disposable_factory=Mistral,
    ),
    "ollama": ProviderSpec(
        key="ollama",
        display_name="Ollama",
        default_model="hf.co/mradermacher/M-Prometheus-3B-i1-GGUF:Q4_K_M",
        config_factory=lambda model: OllamaChatConfig(
            model=model,
            temperature=0,
            max_output_tokens=512,
        ),
        disposable_factory=Ollama,
    ),
}


def _parse_provider(value: str) -> ProviderSpec:
    key: str = value.strip().lower()
    if key not in PROVIDERS:
        available: str = ", ".join(spec.display_name for spec in PROVIDERS.values())
        raise argparse.ArgumentTypeError(f"Unsupported provider '{value}'. Available: {available}")

    return PROVIDERS[key]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a draive LLM evaluator against a human-labeled baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        type=_parse_provider,
        help="LLM provider to use as the evaluator judge (anthropic|openai|gemini|mistral|ollama)",
        metavar="PROVIDER",
    )
    parser.add_argument(
        "--model",
        help="Override the default model for the chosen provider",
    )
    parser.add_argument(
        "--baseline",
        help="Path to the baseline YAML file",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of concurrent evaluator invocations (default: 4)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip failing samples and report them at the end instead of aborting",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List evaluators that have registered baseline support and exit",
    )
    return parser


async def _verify(
    provider: ProviderSpec,
    document: BaselineDocument,
    /,
    *,
    model: str | None,
    concurrency: int,
    continue_on_error: bool,
) -> SuiteResult:
    return await run_suite(
        document,
        config_state=(provider.create_config(model),),
        disposable_factories=(provider.create_disposable,),
        concurrency=concurrency,
        continue_on_error=continue_on_error,
    )


def _list_evaluators() -> None:
    sys.stdout.write("Registered evaluators:\n")
    for name in available_evaluators():
        entry = lookup_evaluator(name)
        sys.stdout.write(f"  - {name}: {entry.description}\n")


def main() -> None:
    load_env()
    setup_logging("evals.verify_evaluator")

    parser: argparse.ArgumentParser = _build_parser()
    args: argparse.Namespace = parser.parse_args()

    if args.list:
        _list_evaluators()
        return

    if args.provider is None or args.baseline is None:
        parser.error("--provider and --baseline are required (use --list to inspect evaluators)")

    document: BaselineDocument = load_baseline(args.baseline)

    # Validate evaluator name early to give a clear error before any API call.
    lookup_evaluator(document.evaluator)

    try:
        result: SuiteResult = asyncio.run(
            _verify(
                args.provider,
                document,
                model=args.model,
                concurrency=args.concurrency,
                continue_on_error=args.continue_on_error,
            )
        )

    except KeyboardInterrupt:
        sys.stderr.write("\nVerification interrupted\n")
        sys.exit(130)

    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)

    sys.stdout.write(result.render() + "\n")

    if result.failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
