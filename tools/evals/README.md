# Evaluator baseline verification

This suite measures how closely draive's LLM-based evaluators agree with
a labeled baseline using Cohen's kappa. The shipped baselines under
`baselines/` are rubric-anchored synthetic samples — see the entry below
for the caveat. Drop in your own human-labeled YAML for a real human
baseline.

## Running

```bash
# List every evaluator that has a registered baseline mapping.
uv run tools/evals/verify_evaluator.py --list

# Verify against OpenAI as the judge.
uv run tools/evals/verify_evaluator.py \
    --provider openai \
    --baseline tools/evals/baselines/coherence.yaml

# Pin a specific model and raise concurrency.
uv run tools/evals/verify_evaluator.py \
    --provider anthropic \
    --model claude-sonnet-4-5 \
    --baseline path/to/coherence_real.yaml \
    --concurrency 8

# Skip failing samples instead of aborting; they appear in the report.
uv run tools/evals/verify_evaluator.py \
    --provider gemini \
    --baseline tools/evals/baselines/relevance.yaml \
    --continue-on-error

# Verify against a local Ollama model, for example M-Prometheus GGUF.
uv run tools/evals/verify_evaluator.py \
    --provider ollama \
    --baseline tools/evals/baselines/fluency.yaml \
    --concurrency 1
```

Supported providers: `anthropic`, `openai`, `gemini`, `mistral`, `ollama`.
Set the matching API key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GOOGLE_API_KEY`, or `MISTRAL_API_KEY`) for hosted providers. For Ollama,
ensure the local server is running and the selected model is available. A local
`.env` next to the script is loaded automatically.

Exit codes:
- `0` — verification ran and every sample succeeded.
- `1` — verification could not run (bad arguments, load error, all
  samples failed, etc.).
- `2` — verification ran but one or more samples failed; see the
  `failures:` block in the rendered report.
- `130` — interrupted (Ctrl-C).

## Reading the report

The runner prints both unweighted Cohen's kappa (nominal) and quadratic
weighted Cohen's kappa (ordinal). For the ordered rating scale used by
draive, the quadratic-weighted variant is usually the headline number —
it penalises near-miss disagreements (`good` vs `excellent`) less
harshly than far-miss ones (`poor` vs `perfect`).

Landis & Koch interpretation:
- `< 0.00` poor
- `0.00–0.20` slight
- `0.21–0.40` fair
- `0.41–0.60` moderate
- `0.61–0.80` substantial
- `0.81–1.00` almost perfect

The agreement itself is computed by draive's own `cohen_kappa_evaluator`
(`draive.evaluators.cohen_kappa_evaluator`) — the runner feeds it the
evaluator scores against the human baseline, so the verification suite and
the shipped framework share a single kappa implementation.
