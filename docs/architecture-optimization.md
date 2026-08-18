# Architecture Optimization Implementation

Date: 2026-08-17

This document records the implemented architecture, rollout controls, and the
evidence available from the local development environment. The implementation
does not rename sources, re-ingest the corpus, re-embed chunks, commit changes,
or require Groq.

## Runtime architecture

1. Simple questions use deterministic two-resource routing (`PDF`, `Excel`, or
   both). PDF narrative is the default. Table and figure support is added only
   by strong cues or one structured uncertain-case judge call.
2. Exact WMP entity-history questions retain the existing typed, validated
   Excel fast path.
3. Complex questions use a typed planner with up to six atomic tasks. Tasks are
   normalized and deduplicated, then capped at four initial retrieval branches.
4. Branches run concurrently, preserve independent evidence groups, and feed a
   deterministic coverage ledger. At most one missing task is retried.
5. Cross-resource arithmetic uses typed fact requirements and allow-listed
   `Decimal` operations. Unit, period, missing-operand, and zero-denominator
   checks run before calculation. Every result carries both operands'
   provenance and can enter the answer prompt as a derived citation.
6. If planning or judging fails, retrieval broadens safely; it does not narrow
   to an unverified model guess.

## Independent rollout flags

Each flag can be set with `SDGE_FEATURE_<NAME>=true|false`; defaults are listed
in `config/config.example.yaml`.

- `TWO_RESOURCE_ROUTER`
- `SUPPORT_EVIDENCE_JUDGE`
- `TYPED_PLANNER`
- `PARALLEL_RETRIEVAL`
- `COVERAGE_RETRY`
- `METADATA_ROUTING`
- `CROSS_RESOURCE_COMPUTATION`
- `PARENT_CHILD_EXPANSION` (off by default)

## Source metadata

`config/source_manifest.json` inventories 11 PDFs and 14 cleaned Excel-table
CSVs. Runtime document-role routing resolves stable roles through this manifest
instead of embedding filenames in routing decisions. The schema adds only a
`source_registry` table; it does not mutate `documents`, `chunks`, facts,
embeddings, citations, or object keys.

Read-only audit:

```bash
uv run python -m scripts.manage_source_manifest
```

Apply the additive registry after PostgreSQL is running and the audit is clean:

```bash
uv run python -m scripts.manage_source_manifest --apply
```

The local file audit found 25/25 sources, zero missing, and zero untracked, so a
metadata-only backfill is sufficient. PostgreSQL was stopped during this task,
therefore `--apply` was intentionally not run.

## Frozen diagnostics

`eval/architecture` contains three 24-case suites:

- Excel execution: lookup, aggregation, null/missing, trends, ranking, and year
  scope cases derived from the existing reviewed Excel challenge set.
- Cross-resource computation: typed Excel/PDF operands, formula, expected
  value, units, periods, provenance, and abstention status.
- Modality gating: eight narrative, eight table, and eight figure cases.

Gold-bearing development files and question-only blind projections are stored
in separate directories. The deterministic builder made zero external calls:

```bash
uv run python -m eval.build_architecture_diagnostics
```

All retrieval evaluators now attach machine, backend, model, memory, context,
concurrency, and software-version metadata to new result artifacts.

## Local planner experiment

Qwen3.5-9B (`Q4_K_M`) was evaluated locally through Ollama/Metal on a 16 GiB
Apple Silicon machine. No Groq calls were made.

| Trial | Structured validity | Required-source fact coverage | p50 | p95 | Result |
|---|---:|---:|---:|---:|---|
| Initial prompt, 8 cases | 0/8 | not scoreable | 16.2 s | 31.2 s | Rejected; field contract ambiguous |
| Corrected exact schema, 8 cases | 8/8 | 14/16 (87.5%) | 18.7 s | 24.2 s | Rejected for coverage and latency |

The exact-schema prompt fix was retained because it removed a real structured
output failure. Qwen3.5-9B was not promoted as the default planner because it
missed the plan's 90% fact-coverage gate and the local 4-second target. The
existing `qwen3:4b` default remains; it was not installed locally, so a control
run was not possible. Cold-start latency is marked unavailable because model
residency was not reset before the retained run. Groq was not used as a
substitute.

## Verification status

- Core unit tests: 105 passed after implementation.
- Backend/API tests: 30 passed after implementation.
- Source manifest audit: 25 found, 0 missing, 0 untracked.
- Groq calls during implementation/evaluation: 0.
- Full retrieval and end-to-end quality gate: not run because local PostgreSQL
  was not running.
- AWS CUDA benchmark: intentionally deferred; local Metal results are not an
  AWS latency claim.

The code is ready for review and deterministic testing. It is not yet approved
for production rollout because the mandatory +8 end-to-end quality gate and
database-backed retrieval checks still require a running corpus database.
