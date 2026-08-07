# Full local pipeline

This document records the no-cost local integration completed on 2026-08-07.
It uses PostgreSQL/pgvector, the Task 1/2 parsers and retrievers, Task 3, FastAPI,
React, and Ollama. It does not call AWS.

## Local corpus and database

Database: `sdge_full_pipeline` in the local `sdge-smoke-pgvector` container.

- 10 PDFs from `files/`
- 2 XLSX workbooks from `files/`
- 14 reviewed QDR CSV files from `excel_cleaning/cleaned_csv_rag_ready/`
- 26 document records
- 3,541 narrative chunks
- 520 Excel cards
- 2 table chunks and 1 figure chunk

All ten PDFs have searchable narrative text. The small Change Order PDF also
completed Docling table/figure extraction. The other nine PDFs used narrative-
only extraction because full Docling extraction of the 1,000-page documents
exceeded the available local Windows memory/time budget. This does not block
narrative retrieval, but tables and figures in those nine PDFs still need a
larger machine or a staged structured-extraction job.

## Ingest all local sources

Run from the repository root in PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
$env:TORCHDYNAMO_DISABLE = "1"
$env:TORCH_COMPILE_DISABLE = "1"
python -m uv run python -X utf8 scripts/ingest_all_local.py --pdf-mode narrative --skip-existing-pdfs
```

The command discovers every PDF/XLSX/XLSM below `files/` and every reviewed
CSV in `excel_cleaning/cleaned_csv_rag_ready/`. The safe local default is
`--pdf-mode narrative`. Use `--pdf-mode structured` only when the machine can
run Docling table/figure extraction for the full corpus.

## Run the application

Prerequisites:

- Docker container `sdge-smoke-pgvector` is running on port 55432.
- Ollama is running and the model named by `OLLAMA_MODEL` is installed.
- `config/config.yaml` points at the local database.
- `.env` contains the database password and local Ollama settings.
- Python and frontend dependencies have been installed with `python -m uv
  sync` and `npm ci`.

Start both services from the repository root:

```powershell
.\scripts\run_local.ps1
```

Then open `http://127.0.0.1:5173/`. FastAPI listens on
`http://127.0.0.1:8000/`; its health endpoint is `/api/health`.

FastAPI preloads the embedding model, reranker, and local Ollama model during
startup. `OLLAMA_KEEP_ALIVE` controls how long Ollama keeps the model resident;
the local development value is 30 minutes. Set `MODEL_WARMUP_ENABLED=false` to
disable startup warmup. Warmup is implemented only for local providers and
does not invoke Bedrock.

React also calls `POST /api/warmup` when the page opens. The input is disabled
and the header displays `preparing models...` until this call succeeds. This
refreshes Ollama residency after a long idle period without changing the
public `/api/ask` contract.

## Runtime request path

1. React sends `{ "question": "..." }` to `POST /api/ask`.
2. FastAPI creates a request ID when the browser does not supply one.
3. `RetrievalService` calls Task 2 grouped retrieval for narrative, table,
   figure, and Excel evidence. It also runs the execution-verified Excel
   channel when the question matches a supported QDR operation.
4. `GenerationService` converts Task 2 `RankedResult` objects to Task 3
   `Chunk` objects. An execution-verified Excel result is inserted as explicit
   evidence when available.
5. `AnswerService` budgets the prompt, calls local Ollama, validates every
   cited chunk ID, and copies citation metadata from the input chunks.
6. FastAPI returns only `request_id`, `answer`, `cited_chunk_ids`,
   `citations`, and `insufficient_context`.
7. React displays the answer and PDF page or Excel sheet/row citation.

Provider failures return HTTP 502 with `answer_generation_failed`. Retrieval
failures return HTTP 503 with `retrieval_failed`. Detailed exception data stays
in backend logs and is not exposed to the browser.

Each completed request writes one `pipeline_timing` log entry containing:

- database connection time;
- grouped Task 2 retrieval time;
- execution-verified Excel channel time;
- Task 2-to-Task 3 adapter time;
- prompt construction time;
- model call time and provider-reported model latency;
- JSON parsing and citation-validation time;
- model input/output token counts and total request time.

These diagnostics are internal and are deliberately omitted from the public
Task 4 response contract.

## Verified live examples

Excel question:

> How many project activities did SDG&E list in its Q4 2024 update?

Result: `20`, supported by execution-verified Table 1 rows 2-16.

PDF question:

> What percentage change in initiative risk requires SDG&E to submit a change
> order report under Energy Safety's 2023-2025 WMP guidelines?

Result: `25%`, citing the Change Order Report, pages 2-3.

Unsupported question:

> What was SDG&E's approved 2035 lunar power generation target on the Moon?

Result: fixed insufficient-evidence message, no citations, and
`insufficient_context=true`.
