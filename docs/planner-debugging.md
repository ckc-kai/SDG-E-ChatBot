# Planner batching and model-call debugging

This change keeps the public question/answer pipeline intact while fixing three
planner failures: dropped activity status, single-result CAPEX/OPEX execution,
and non-reproducible model debugging.

## Retrieval planning

The top-level retrieval planner still returns `RetrievalPlan` and falls back to
one broad, all-content retrieval step when model output is invalid. Complex
questions now receive a 1,200-token primary planner budget. A failed primary
plan may be retried once with the configured Groq escalation model and a
1,500-token budget.

The prompt distinguishes requested results from retrieval work. Calculations
remain requirements, but retrieval tasks ask for their source operands. Metrics
that share one Excel entity and reporting period are batched into one retrieval
task so that CAPEX and OPEX do not compete as duplicate top-level branches.
Optional plan fields must be omitted when unknown; prompt examples do not carry
real activity identifiers or placeholder filter values.

## Excel planner batches

The internal Excel model schema now returns a top-level `plans` array. Each item
is independently sanitized, grounded against known table metadata, executed,
and checked for usable evidence. This permits CAPEX and OPEX to use separate
typed filters while remaining one top-level retrieval step.

Compatibility is preserved:

- `build_model_plan(...)` remains available and returns the first validated
  plan/result pair.
- `answer_from_excel(...)` keeps its existing default single-answer return.
- Internal retrieval orchestration opts into `multiple=True` and preserves all
  validated Excel answers for generation.

For activity-history questions that request status, the typed `status` column
is selected and carried into the verified evidence chunks. Missing status is a
decline condition when the user explicitly requested it.

## Deterministic seeds

`MODEL_SEED` defaults to `42`. `OLLAMA_SEED` and `GROQ_SEED` may override it for
one provider. Ollama and Groq receive the seed in their request payloads.
DeepSeek and Bedrock do not receive a seed because their current provider
interfaces do not support one; their traces record `seed: null`, temperature,
model metadata, and token usage instead.

Seeds improve reproducibility but do not guarantee identical hosted-model
responses after a provider model or serving stack changes.

## Full model traces

Set an opt-in directory before starting the backend:

```dotenv
MODEL_TRACE_DIR=/absolute/private/path/model-traces
```

Every Ollama, Groq, DeepSeek, and Bedrock generation call writes one JSON file
containing the prompt, structured schema when present, request payload without
authorization headers, raw model text, usage, response metadata, elapsed time,
outcome, and a prompt SHA-256 hash.

These files can contain user questions, retrieved evidence, and model answers.
Keep the directory outside the repository, restrict access, and do not commit
the artifacts.

## Planner traces and replay

Planner-specific traces add the planning trigger, accepted normalized plan, or
rejection reason:

```dotenv
TASK3_PLANNER_TRACE_DIR=/absolute/private/path/planner-traces
```

To reproduce planner parsing and validation without calling the model again,
point to exactly one planner trace:

```dotenv
TASK3_PLANNER_REPLAY_FILE=/absolute/private/path/planner-traces/planner-....json
```

Replay is rejected if either the provider model ID or prompt hash differs. This
prevents a cached response from silently being evaluated against another model
or a changed prompt.

## Verification

Run the focused regression suite from the repository root:

```bash
.venv/bin/python -m unittest \
  tests.test_planning \
  tests.test_excel_plan_batch \
  tests.test_query_scope \
  tests.test_ollama_provider \
  tests.test_groq_provider \
  tests.test_deepseek_provider \
  tests.test_bedrock_provider

PYTHONPATH=backend:. .venv/bin/python -m unittest \
  backend.tests.test_generation_service \
  backend.tests.test_retrieval_service
```

The 27-question planner-only benchmark is a semantic diagnostic rather than a
release gate. The pre-cleanup Qwen3:4B run produced structurally valid plans for
all 25 model-planned questions, but exact expected-source routing matched only
14 of 27 questions. In particular, none of ten combined PDF/Excel questions
covered both sources. This branch removes observed prompt-value leakage, but it
does not add the semantic source-coverage validator needed to make those model
plans production-safe.
