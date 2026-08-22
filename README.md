# SDG-E-ChatBot
UCLA MEng Capstone project for grounded question answering over SDG&E
regulatory filings.

Task 2 retrieval code is in `retrieval/`. Task 3's framework-neutral generation
core is in `generation/`; it supports deterministic mocks, local Ollama,
DeepSeek, Groq, and an Amazon Bedrock Converse provider. Live Bedrock calls remain
disabled until account access and credentials are available. See
`docs/task3-contract.md` for the Task 2/Task 4 contract and
`docs/task3-ollama.md` for local end-to-end testing.

## Local web app quick start

The database must already be running and populated. Starting the web app does
not ingest the corpus again.

```powershell
# One-time setup
uv sync
Copy-Item .env.example .env
Set-Location frontend
npm ci
Set-Location ..

# Local answer model (in a separate terminal)
ollama pull qwen3:4b
ollama serve

# Start FastAPI and React
powershell -ExecutionPolicy Bypass -File scripts/run_local.ps1
```

Set `TASK3_PROVIDER=ollama` and `OLLAMA_MODEL=qwen3:4b` in `.env`, then open
<http://127.0.0.1:5173>. Backend/frontend logs are written under `logs/local/`.

# Excel Cleaning

This directory contains scripts for cleaning and standardizing the tabular data used in this project.

## Structure

* **`excel_cleaning/`** – Contains Python scripts for cleaning the source Excel tables.

  * Each script corresponds to a specific table and produces a cleaned version of that table.
* **`excel_cleaning/cleaned_csv_rag_ready/`** – Contains the cleaned CSV files that are ready for ingestion into the RAG pipeline.


# SDG&E WMP Retrieval

This project ingests SDG&E Wildfire Mitigation Plan PDFs and retrieves
supporting narrative text, tables, and figures through one PostgreSQL/pgvector
index.

The supported workflow has:

- separate PDF and Excel ingest commands;
- one `chunks` table for every content type;
- raw and contextual embeddings generated automatically for every chunk;
- separate PDF and execution-verified Excel query paths;
- independently ranked narrative, PDF-table, PDF-figure, and Excel evidence;
- no query-type classifier and no cross-content score calibration.

## Retrieval architecture

```text
PDF
 ├─ bookmark-aware narrative extraction ─┐
└─ Docling table/figure extraction ─────┤
                                         ├─ unified chunks table ─┐
Excel ── contract-driven facts/cards ────┘                        │
                                                                  │
question ── retrieve every requested evidence group
 ├─ narrative group
 ├─ PDF-table group
 ├─ PDF-figure group
 └─ Excel-card group

Each group
 ├─ raw vector search
 ├─ contextual vector search
 ├─ PostgreSQL lexical search
 └─ candidate merge → rerank inside the group → return with provenance
```

Reranker scores are comparable only inside a group. Grouped retrieval prevents
an Excel card from displacing narrative evidence, or a figure from displacing
a table. 

Tables retain their exact grid in `structured_data`. Figure crops are stored
through a filesystem/S3 abstraction and referenced by `object_key`.

Generated figure descriptions are deliberately separated from authoritative
content:

- `content` contains caption and deterministic PDF page context;
- `retrieval_hint` may contain the local vision-model description;
- the hint can improve candidate recall;
- by default, it is excluded from reranking and answer evidence.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- PostgreSQL with the pgvector extension
- source PDFs under `resources/wmp/pdf/`

Install the project:

```bash
uv sync
cp config/config.example.yaml config/config.yaml
cp .env.example .env
```

Set the local database password in `.env`:

```dotenv
POSTGRES_PASSWORD=change-me
```

`config/config.yaml`, `.env`, source PDFs, and extracted figure files are
ignored by Git.

## Run locally

The refactor uses a new unified schema. The following reset command permanently
deletes the previously ingested `documents`, `chunks`, `structured_chunks`, and
structured ingest state. It does not delete the source PDFs.

```bash
uv run python -m retrieval.setup_db --reset --yes
```

Ingest every PDF, including narrative text, tables, and figures:

```bash
uv run python -m retrieval.ingest.pdf
```

Ingest only filenames containing a substring:

```bash
uv run python -m retrieval.ingest.pdf --only "2023_Base-WMP"
```

Ingest the cleaned Excel CSVs:

```bash
uv run python -m retrieval.ingest.excel
```

Query all content types as independently ranked evidence groups:

```bash
uv run python -m retrieval.query.pdf \
  "What is the 2024 updated target for the Strategic Pole Replacement program?"
```

The output labels every result as `narrative`, `table`, `figure`, or
`excel_card`. Figure results also print the resolved local path or S3 URI.
Pass `--legacy-flat` only when comparing with the previous global mixed ranking.

Application code should honor the configured output contract through the
dispatcher (or call `retrieve_evidence` explicitly):

```python
from retrieval.query.pdf import retrieve_configured

evidence = retrieve_configured(question, connection)
```

The low-level `retrieve()` function intentionally keeps its historical flat
list return type for source compatibility.

Automatic query decomposition is controlled by `query_rewrite.mode`. Set it to
`off` to guarantee no Anthropic request:

```yaml
query_rewrite:
  mode: off
```

## Configuration

The checked-in [configuration example](config/config.example.yaml) contains
all non-secret choices:

- PostgreSQL deployment and connection behavior;
- embedding/reranker provider, model, and dimensions;
- chunk sizes;
- hybrid retrieval and reranking controls;
- Docling table/figure extraction;
- generated figure-hint policy;
- filesystem or S3 object storage.

Set `SDGE_CONFIG_PATH` to use a YAML file at another path:

```bash
export SDGE_CONFIG_PATH=/path/to/config.yaml
```

### Switch models

The currently implemented model provider is `sentence_transformers`. Change
the embedding or reranker model in YAML:

```yaml
models:
  embedding:
    provider: sentence_transformers
    name: BAAI/bge-base-en-v1.5
    dimensions: 768
  reranker:
    provider: sentence_transformers
    name: BAAI/bge-reranker-base
```

Changing the embedding model triggers a clean vector refresh on the next
ingest. For a model with different dimensions, build a second database, ingest
and evaluate it, then switch the application connection:

```bash
createdb sdge_next_embedding
uv run python -m retrieval.setup_db
uv run python -m retrieval.ingest.pdf
uv run python -m retrieval.ingest.excel
```

A pgvector column has a fixed dimension, so a dimension change cannot be
applied safely to the active index through configuration alone. A parallel
database keeps the current collection available for immediate rollback locally
and on RDS/Aurora. Changing only the reranker does not require re-ingestion.

The provider factory rejects unsupported values clearly. It is ready for a
future Bedrock or SageMaker adapter, but this refactor intentionally implements
only local SentenceTransformers models.

### Configure generated figure hints

```yaml
extraction:
  structured:
    figure_description:
      generate: false
      candidate_retrieval: false
      reranking: false
      answer_context: false
```

The lightweight default avoids a multi-GB vision-model dependency and relies
on captions, page context, and the stored crop. Vision hints can be enabled as
an optional recall experiment; they never become reranking or answer evidence.

To remove the vision model from ingest entirely:

```yaml
extraction:
  structured:
    figure_description:
      generate: false
      candidate_retrieval: false
      reranking: false
      answer_context: false
```

Retrieval then relies on captions, section breadcrumbs, page context, and the
stored figure crop.

## Migrate to AWS

The production target is Amazon RDS for PostgreSQL with pgvector. Aurora
PostgreSQL remains a compatible later option; both use the same application
code and `psycopg2` connection path.

Before selecting an engine version, check AWS's current
[RDS PostgreSQL extension matrix](https://docs.aws.amazon.com/AmazonRDS/latest/PostgreSQLReleaseNotes/postgresql-extensions.html).
AWS also documents
[pgvector setup for Aurora PostgreSQL](https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/AuroraPostgreSQL.VectorDB.html).

### RDS PostgreSQL

Create an RDS PostgreSQL instance whose engine version supports pgvector, make
it reachable from the application VPC/security group, and change only the
database and storage settings:

```yaml
database:
  provider: postgresql
  deployment: rds
  host: your-instance.region.rds.amazonaws.com
  port: 5432
  name: postgres
  user: app_user
  password_env: POSTGRES_PASSWORD
  sslmode: require
  connect_timeout_seconds: 10

object_storage:
  provider: s3
  s3:
    bucket: your-private-bucket
    prefix: sdge-chatbot/figures
    region: us-west-2
```

Do not place the password or AWS access keys in YAML. In AWS, inject the
database password from Secrets Manager into `POSTGRES_PASSWORD` and use an IAM
role for S3 access. The role needs object list, write, and delete permissions
scoped to the configured bucket prefix.

The S3 adapter loads `boto3` only when `provider: s3` is selected. Install the
locked AWS extra in the ingest-worker image; local filesystem runs do not need
these packages:

```bash
uv sync --extra aws
```

Run the same commands from the AWS ingest worker:

```bash
uv run python -m retrieval.setup_db
uv run python -m retrieval.ingest.pdf /path/to/downloaded/pdfs
```

The setup command creates the `vector` extension, unified schema, HNSW vector
indexes, and PostgreSQL full-text index. For production, give the schema-setup
identity permission to create extensions/tables; the steady-state ingest/query
identity can use narrower privileges.

### Switch RDS to Aurora later

No retrieval or ingest code changes are required:

```yaml
database:
  provider: postgresql
  deployment: aurora
  host: your-cluster.cluster-region.rds.amazonaws.com
  port: 5432
  name: postgres
  user: app_user
  password_env: POSTGRES_PASSWORD
  sslmode: require
```

Point the host at the Aurora writer endpoint for ingestion. Query-only workers
may use an appropriate reader endpoint after confirming read-after-write and
index availability requirements.

## Evaluation after the clean re-ingest

This code refactor does not migrate or re-ingest the existing local database.
After the clean ingest, run both evaluation sets through the same unified query
implementation.

The unified evaluator auto-detects the suite and writes a dated result file.
Use `--grouped` for the production evidence-group design. Omit it only to
measure the current cross-content merged route.

Narrative evaluation:

```bash
uv run python -m eval.run_eval \
  --input eval/pdf/evaluation_natural.jsonl \
  --pdf narrative \
  --grouped \
  --output current-route-gate \
  --misses
```

Table/figure evaluation:

```bash
uv run python -m eval.run_eval \
  --input eval/pdf/evaluation_structured.jsonl \
  --pdf structured \
  --grouped \
  --output current-route-gate \
  --misses
```

Excel evaluation:

```bash
uv run python -m eval.run_eval \
  --input eval/excel/evaluation_excel.jsonl \
  --excel \
  --grouped \
  --output current-route-gate
```

Results are written to `eval/pdf/results/narrative/`,
`eval/pdf/results/structured/`, or `eval/excel/results/` as
`YYYY-MM-DD-FEATURE.json`. Add `--oracle` to evaluate only the suite's own lane.
Historical pre-unification results remain in the PDF result subfolders, but
should not be treated as measurements of the new unified schema.

## Project layout

```text
config/                         Runtime settings and reviewed Excel contracts
retrieval/
├── ingest/
│   ├── pdf/                    PDF narrative/table/figure ingestion
│   └── excel/                  Excel contracts, transforms, cards, and ingest
├── query/
│   ├── pdf/                    Narrative and structured-PDF retrieval
│   ├── excel/                  Excel planning and execution-verified answers
│   ├── lanes.py                Lane definitions and confidence signals
│   └── calibration.py          Optional diagnostic score calibration
├── contextual_embeddings.py   Shared embedding recipe
├── object_storage.py          Filesystem and S3 figure storage
├── setup_db.py                Base PostgreSQL/pgvector schema
└── utils.py                   Shared configuration, clients, and DB connection
eval/
├── pdf/                       PDF evaluation data and historical results
└── excel/                     Excel evaluation data
```
### Use Groq after creating a Free-plan account

Create an API key in the Groq console, keep the account on the **Free** plan,
and put the following values in your local `.env` (never commit the real key):

```dotenv
TASK3_PROVIDER=groq
GROQ_API_KEY=gsk_your_local_key
GROQ_MODEL=openai/gpt-oss-120b
MODEL_WARMUP_ENABLED=false
```

Then start the backend and frontend using the quick-start commands above. No Groq SDK
is required. The prompt requests JSON; Task 3 validates its schema and cited chunk IDs, and copies
citation metadata from the retrieved chunks. A free-tier `429` is returned as a
controlled provider error and is not automatically retried. The application
cannot control the account's billing plan, so verify that the Groq console says
**Free** and do not add/upgrade to a paid plan if zero cost is required.

### Use DeepSeek for answer-generation comparison

Create a DeepSeek API key and put these values in the local `.env` (never
commit the real key):

```dotenv
TASK3_PROVIDER=deepseek
TASK3_PLANNER_PROVIDER=ollama
TASK3_PLANNER_MODEL=qwen3.5:9b
TASK3_PLANNER_MAX_TOKENS=1200
DEEPSEEK_API_KEY=your_local_key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_THINKING=disabled
DEEPSEEK_PROMPT_TOKEN_BUDGET=6500
DEEPSEEK_MAX_TOKENS=1500
MODEL_WARMUP_ENABLED=false
```

This setup uses the local `qwen3.5:9b` model only for question rewrite/planning
and DeepSeek V4 Flash for final answer generation. The default evidence and
output budgets match the Groq comparison. DeepSeek's JSON-object output is
parsed and citation-validated by the same Task 3 code; retrieval, citation
hydration, and the public FastAPI response do not change.
DeepSeek API calls are billed to the configured DeepSeek account, so review the
provider's current pricing and balance before running live tests.

Planner batching, deterministic seeds, model-call traces, and planner replay are
documented in [`docs/planner-debugging.md`](docs/planner-debugging.md).
