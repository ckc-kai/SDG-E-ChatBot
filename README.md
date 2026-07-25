# SDG&E WMP Text Retrieval

This project retrieves supporting text from SDG&E Wildfire Mitigation Plan
(WMP) PDFs. It uses PostgreSQL with pgvector, BGE embeddings, and a cross-encoder
reranker.

The accepted retrieval pipeline is the raw-preserving hybrid union:

```text
question
   ├─ raw embedding search ───────── top 30 ──┐
   └─ contextual embedding search ─ top 30 ──┤
                                              ├─ deduplicated union
                                              └─ BGE reranker ── top 10
```

All raw candidates are retained. Candidates found only by contextual retrieval
are appended before the combined pool is reranked. This prevents the contextual
channel from evicting strong raw candidates while preserving its complementary
recall.

## Current models

- Embedding: `BAAI/bge-base-en-v1.5`
- Reranker: `BAAI/bge-reranker-base`
- Vector dimensions: 768
- Raw candidates per query: 30
- Contextual candidates per query: 30
- Final reranked results: 10

Each stored chunk has two embeddings:

1. **Raw embedding** — the stored chunk content.
2. **Contextual embedding** — an embedding-only representation:

   ```text
   Document: <source PDF filename>
   Section: <full breadcrumb>
   Chunk: <stored chunk content>
   ```

The stored/displayed content is not modified. If the complete contextual input
exceeds the embedding model limit, only the supplementary `Document:` line is
removed; the full breadcrumb and chunk content are preserved.

## Evaluation result

The accepted system was evaluated on the 150-question natural dataset:

`eval/pdf/results/2026-07-23-a3b-hybrid-union-natural.json`

| Metric | Score |
|---|---:|
| Hit@1 | 0.8533 |
| Recall@5 | 0.9400 |
| MRR@5 | 0.8919 |
| nDCG@5 | 0.9006 |
| Recall@10 | 0.9633 |
| MRR@10 | 0.8945 |
| nDCG@10 | 0.9084 |

The deduplicated reranker pool contained 32–53 chunks per question, with a mean
of 41.83. Candidate generation missed gold evidence for 3 questions, and the
reranker missed gold evidence in its top 10 for 2 additional questions.

Historical ablation outputs remain under `eval/pdf/results/` for reproducibility,
but they are not part of the supported retrieval workflow.

## Setup

Requirements:

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- PostgreSQL with the pgvector extension

Install the Python dependencies:

```bash
uv sync
```

Create a local configuration:

```bash
cp config/config.example.yaml config/config.yaml
```

Update the database credentials in `config/config.yaml`. If automatic query
decomposition is enabled, also copy `.env.example` to `.env` and provide an
Anthropic API key.

Place the source PDFs in:

```text
resources/wmp/pdf/
```

The `resources/` directory and local configuration are intentionally ignored by
Git.

## Ingest PDFs

Run:

```bash
uv run python -m retrieval.ingest
```

Ingestion now performs the complete database preparation and embedding workflow:

1. Creates or updates the PostgreSQL/pgvector schema.
2. Extracts bookmark-based leaf sections from each PDF.
3. Builds stable, overlapping token chunks.
4. Generates both raw and contextual embeddings.
5. Stores both vectors with the chunk in one transaction.

No separate contextual-embedding backfill command is required:

- New or changed PDFs are chunked and receive both embeddings.
- Unchanged PDFs with current embeddings are skipped.
- Unchanged PDFs with missing or stale contextual embeddings are updated in
  place without deleting chunks or changing chunk IDs.

## Structured ingest (tables, charts, figures)

An additive second pass extracts tables and figures with
[docling](https://github.com/docling-project/docling) (local layout +
TableFormer models, no cloud calls) into a separate `structured_chunks` table.
The narrative `chunks` table, its ingest, and the accepted text pipeline are
untouched.

```bash
uv run python -m retrieval.structured_ingest
```

What it stores per element:

- **Tables** — a Markdown flattening in `content` (embedded and reranked like
  any chunk, split row-aware with the header repeated in every piece) plus the
  exact cell grid as JSONB in `structured_data` for precise cell quoting.
- **Figures/charts** — caption + a local SmolVLM-256M description + the
  figure's own page text as `content`, and the cropped PNG under
  `resources/wmp/figures/` (`image_path`).

Both carry the same breadcrumb / sub-document metadata as narrative chunks
(mapped by page through the same bookmark leaf-section tree) and both raw and
contextual embeddings. Noise filters drop table-of-contents "tables"
(dot-leader density) and captionless branding artwork.

Re-runs skip unchanged PDFs; changing `structured.*` config or the extractor
version re-extracts automatically (`structured_ingest_state`).

## Query across narrative + structured chunks

```bash
uv run python -m retrieval.structured_query \
  "What is the 2024 updated target for the Strategic Pole Replacement program?"
```

Same accepted pipeline shape (hybrid raw+contextual, raw-preserving union,
cross-encoder rerank), but each dense search UNIONs `chunks` with
`structured_chunks`. Structured results are printed with a `[table]`/`[figure]`
marker and the figure image path when present.

## Structured evaluation

Generate a table/figure question set from ingested structured chunks (drafted
with the same Claude Haiku model the query-rewrite path uses), then score it:

```bash
uv run python -m eval.generate_structured_eval --tables 30 --figures 15
uv run python -m eval.run_structured_eval --misses \
  --out eval/pdf/results/structured-eval.json
```

Gold references use the stable key
(source_pdf, content_type, page_start, chunk_index) against
`structured_chunks`, so results survive re-extraction.

## Query

The example configuration selects the accepted hybrid-union pipeline by
default:

```bash
uv run python -m retrieval.query \
  "What is the purpose of the WiNGS-Planning model?"
```

The equivalent explicit command is:

```bash
uv run python -m retrieval.query \
  --embedding-mode hybrid \
  --hybrid-pool-mode union \
  "What is the purpose of the WiNGS-Planning model?"
```

Automatic query decomposition is controlled by `query_rewrite.mode` in
`config/config.yaml`. Use `off` when reproducing the accepted evaluation.

## Reproduce the accepted evaluation

The full evaluation is intentionally run manually:

```bash
uv run python -m eval.run_eval \
  --eval eval/pdf/evaluation_natural.jsonl \
  --embedding-mode hybrid \
  --hybrid-pool-mode union \
  --rewrite-mode off \
  --retrieval-top-k 30 \
  --rerank-top-k 10 \
  --metric-k 10 \
  --out eval/pdf/results/a3b-hybrid-union-natural-recheck.json \
  --misses
```

## Tests

```bash
uv run python -m unittest discover -s tests
```

## Project layout

```text
config/
  config.example.yaml       Reproducible non-secret configuration template
eval/
  pdf/evaluation_natural.jsonl
  pdf/results/              Evaluation artifacts
  run_eval.py               Retrieval evaluation harness
retrieval/
  contextual_embeddings.py Contextual embedding input recipe
  ingest.py                 Schema setup, chunking, and both embeddings
  query.py                  Hybrid retrieval, union pooling, and reranking
  setup_db.py               PostgreSQL/pgvector schema
  structured_ingest.py      Docling table/figure extraction and embedding
  structured_query.py       Retrieval across narrative + structured chunks
  structured_schema.py      structured_chunks / structured_ingest_state schema
  utils.py                  Configuration, database, and model loaders
eval/
  generate_structured_eval.py  Table/figure eval dataset generator
  run_structured_eval.py       Structured retrieval scoring harness
tests/
  test_ingest_contextual_embeddings.py
  test_query_fusion.py
  test_structured_ingest.py
```

## Current scope

The accepted text-retrieval evaluation covers narrative chunks only. Tables,
charts, and figures are additionally ingested by the structured pass above and
evaluated separately (`eval/run_structured_eval.py`); flattened element text
may still appear inside narrative chunks, which mildly aids recall and is
accepted.
