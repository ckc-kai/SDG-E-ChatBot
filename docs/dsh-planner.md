# DeepSeek Harness retrieval planner

## Scope

DeepSeek Harness (DSH) is an optional planner inside the existing backend. It
does not replace retrieval, verified Excel execution, answer generation, the
API contract, or the React frontend. The frontend therefore keeps the current
layout and response format.

The intended production routing is:

```text
question
  -> deterministic needs_planning()
     -> false: original rules + optional lightweight auxiliary judge
               -> existing retrieval -> configured answer model
     -> true:  DSH -> typed RetrievalPlan
               -> existing retrieval/execution -> configured answer model
```

`needs_planning()` is implemented in `generation/planning.py`. It is a
deterministic rule gate for multiple questions, comparisons, audits, compound
tasks, multi-document scopes, side-by-side reports, cross-source
reconciliation, longitudinal reported metrics, and explicit references to
multiple workbook tables. It does not call a model.

DSH receives only the user question, the planner instructions, and the JSON
schema for `RetrievalPlan`. It does not retrieve tables or answer factual
questions. The application's existing PDF/Excel retrievers and validated
executors consume the typed plan. This boundary preserves provenance and keeps
DSH from becoming a second, unvalidated retrieval stack.

## Runtime layout

The application remains on Python 3.12. DSH SDK `0.1.1rc1` is loaded by an
isolated Python 3.10 subprocess from `.dsh-python/`; that directory and DSH
session output are ignored by Git. `generation/providers/dsh.py` owns the
application-side adapter, `scripts/run_dsh_planner.py` owns the isolated SDK
process, and `dsh/cordis.yml` defines the minimal runtime.

The checked-in Cordis configuration has no Bash or subprocess executor. A
filesystem service is required by the DSH runtime, but the planner instruction
explicitly forbids workspace inspection. All factual access stays in the
application's retrievers.

For local Qwen3 models, the runner starts a loopback adapter that translates
DSH's DeepSeek-compatible streaming request to Ollama's native `/api/chat`
request with `think=false`. This prevents hidden reasoning output from
consuming the structured-plan budget.

## Installation

From the repository root on the GPU host:

```bash
python3 --version  # expected: Python 3.10.x
python3 -m pip install --target .dsh-python -r dsh/requirements.txt
ollama pull qwen3:14b
```

Do not add the DSH SDK to the application's `uv` environment. Keeping it
isolated avoids changing the Python 3.12 dependency graph.

## Configuration

Copy `.env.example` to `.env` and use:

```dotenv
# Final answer generation
TASK3_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-v4-flash

# needs_planning=false: original lightweight helpers
TASK3_AUXILIARY_PROVIDER=ollama
TASK3_AUXILIARY_MODEL=qwen3:4b
TASK3_AUXILIARY_MAX_TOKENS=500

# needs_planning=true: DSH typed planner
TASK3_PLANNER_PROVIDER=dsh
DSH_PLANNER_MODEL=qwen3:14b
DSH_PLANNER_MAX_TOKENS=1800
DSH_PLANNER_TIMEOUT_SECONDS=180
DSH_BASE_URL=http://127.0.0.1:11434/v1
DSH_API_KEY=ollama
DSH_PYTHON=python3
DSH_SDK_PATH=.dsh-python
```

The `DSH_*` endpoint and key are intentionally separate from
`DEEPSEEK_BASE_URL` and `DEEPSEEK_API_KEY`. This allows the final answer to use
the hosted DeepSeek API while DSH plans with local Qwen3.

To roll back without a code change, set:

```dotenv
TASK3_PLANNER_PROVIDER=ollama
```

## Failure behavior

Invalid DSH output, a missing SDK, timeout, or subprocess failure is converted
to a provider error. Planning then returns the existing broad fallback plan.
If `GROQ_API_KEY` is configured, the existing one-shot hosted planner
escalation may run before the fallback is accepted. The public API schema and
frontend behavior do not change.

## Verification

Run the focused tests:

```bash
PYTHONPATH=.:backend /opt/sdge-chatbot/.venv/bin/python -m unittest \
  tests.test_dsh_provider \
  backend.tests.test_generation_service \
  tests.test_planning
```

The full beta runner uses the same production gate through
`eval/run_full_beta.py`. Runtime settings are captured in its output without
recording API keys. See `docs/experiments/dsh_planner_evaluation.md` for the
latest controlled comparison.
