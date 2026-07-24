# SDG-E-ChatBot

UCLA MEng Capstone project for grounded question answering over SDG&E
regulatory filings.

Task 2 retrieval code is in `retrieval/`. Task 3's framework-neutral generation
core is in `generation/`; it currently uses deterministic local mocks and does
not call AWS or any external answer-generation model. See
`docs/task3-contract.md` for the Task 2 adapter and Task 4 handoff contract.

## A2: contextual embedding ablation

The contextual experiment preserves the existing chunks and raw embeddings. It
adds an embedding-only representation with this format:

```text
Document: <source PDF filename>
Section: <full breadcrumb>
Chunk: <stored chunk content>
```

The stored/displayed chunk content is unchanged. The default retrieval mode is
still `raw`. If the complete contextual input would exceed the embedding
model's token limit, the supplementary `Document:` line is omitted for that
chunk; the full breadcrumb and complete chunk content are preserved.

### 1. Add the columns and backfill existing chunks

The backfill performs the schema migration automatically and updates chunks in
place; it does not re-ingest PDFs or change chunk ids.

```bash
uv run python -m retrieval.backfill_contextual_embeddings
```

The command is resumable. It skips rows already generated with the configured
embedding model and contextual-format recipe. Use `--force` only when you
intentionally want to rebuild every contextual vector.

### 2. Run the contextual evaluation

```bash
uv run python -m eval.run_eval \
  --eval eval/pdf/evaluation_natural.jsonl \
  --embedding-mode contextual \
  --rewrite-mode off \
  --retrieval-top-k 30 \
  --metric-k 10 \
  --out eval/pdf/results/2026-07-22-a2-contextual-off-metric10-candidates30.json \
  --misses
```

This is directly comparable to the corrected raw baseline:

`eval/pdf/results/2026-07-22-off-metric10-candidates30-label-audit.json`

To rerun the raw control with the current code, use the same command with
`--embedding-mode raw` and a different output filename.

### 3. Try a single contextual query

```bash
uv run python -m retrieval.query \
  --embedding-mode contextual \
  "What is the purpose of the WiNGS-Planning model?"
```

## A3: raw + contextual RRF ablation

Hybrid mode retrieves the configured candidate count independently from the raw
and contextual columns, combines their ranks with reciprocal-rank fusion, keeps
the same configured candidate count, and sends that fixed-size pool to the
existing reranker:

```text
raw top 30 --------\
                    RRF (k=60) -> fused top 30 -> existing reranker
contextual top 30 -/
```

Run the A3 evaluation with:

```bash
uv run python -m eval.run_eval \
  --eval eval/pdf/evaluation_natural.jsonl \
  --embedding-mode hybrid \
  --rrf-k 60 \
  --rewrite-mode off \
  --retrieval-top-k 30 \
  --rerank-top-k 10 \
  --metric-k 10 \
  --out eval/pdf/results/2026-07-22-a3-hybrid-rrf60.json \
  --misses
```

The output records the fused candidate ids in `vector_candidate_ids` and the
underlying raw/contextual rankings in `channel_candidate_ids`.

## A3b: raw-preserving contextual union

A3b keeps all 30 candidates from the champion raw channel, appends candidates
that appear only in the contextual top 30, deduplicates by chunk id, and sends
the resulting pool to the existing reranker:

```text
raw top 30 --------------------\
                                deduplicated union (up to 60) -> existing reranker
contextual-only from top 30 ---/
```

Unlike A3's fixed-size equal-weight RRF, contextual retrieval cannot evict a
raw candidate. The only experimental change from A3 is the hybrid candidate
pool policy; embeddings, query rewriting, and reranker model remain unchanged.
`--retrieval-top-k 30` applies independently to each channel, while
`--rerank-top-k 10` controls the final returned ranking rather than the number
of candidates scored by the reranker.

Run A3b on the natural dataset with:

```bash
uv run python -m eval.run_eval \
  --eval eval/pdf/evaluation_natural.jsonl \
  --embedding-mode hybrid \
  --hybrid-pool-mode union \
  --rewrite-mode off \
  --retrieval-top-k 30 \
  --rerank-top-k 10 \
  --metric-k 10 \
  --out eval/pdf/results/2026-07-23-a3b-hybrid-union-natural.json \
  --misses
```

The result diagnostics include the hybrid pool mode and the minimum, maximum,
and mean deduplicated reranker-pool sizes. A3 remains reproducible because
`--hybrid-pool-mode rrf` is still the default.
