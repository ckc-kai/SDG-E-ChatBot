# SDG-E-ChatBot

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
- optional independent narrative, structured-PDF, and Excel retrieval lanes;
- no query-type classifier and no structured/narrative result quota.

## Retrieval architecture

```text
PDF
 ├─ bookmark-aware narrative extraction ─┐
└─ Docling table/figure extraction ─────┤
                                         ├─ unified chunks table ─┐
Excel ── contract-driven facts/cards ────┘                        │
                                                                  │
lane-enabled question                                             │
 ├─ narrative lane ────────────────────────────────────────────────┤
 ├─ structured PDF lane ───────────────────────────────────────────┤
 └─ Excel-card lane ───────────────────────────────────────────────┘

Each lane
 ├─ raw vector search
 ├─ contextual vector search
 ├─ PostgreSQL lexical search
 └─ candidate merge → rerank within lane → cross-lane merge
```

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

Query all content types:

```bash
uv run python -m retrieval.query.pdf \
  "What is the 2024 updated target for the Strategic Pole Replacement program?"
```

The output labels every result as `narrative`, `table`, `figure`, or
`excel_card`. Figure results also print the resolved local path or S3 URI.

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
ingest. If the new embedding model has a different dimension, update
`dimensions`, reset the schema, and re-ingest:

```bash
uv run python -m retrieval.setup_db --reset --yes
uv run python -m retrieval.ingest.pdf
```

A pgvector column has a fixed dimension, so a dimension change cannot be
applied safely to existing vectors through configuration alone. Changing only
the reranker does not require re-ingestion.

The provider factory rejects unsupported values clearly. It is ready for a
future Bedrock or SageMaker adapter, but this refactor intentionally implements
only local SentenceTransformers models.

### Configure generated figure hints

```yaml
extraction:
  structured:
    figure_description:
      generate: true
      candidate_retrieval: true
      reranking: false
      answer_context: false
```

The recommended defaults generate a local SmolVLM hint and use it only to
expand candidate recall. A hallucinated hint therefore cannot directly decide
the reranker order or become factual answer context.

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

The S3 adapter loads `boto3` only when `provider: s3` is selected. Add `boto3`
to the AWS runtime image or deployment dependencies; local filesystem runs do
not require it.

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

The unified evaluator auto-detects the suite from the input JSONL, runs the
current three-lane route by default, and writes a dated result file. Use
`--pdf` or `--excel` to make the adapter explicit.

Narrative evaluation:

```bash
uv run python -m eval.run_eval \
  --input eval/pdf/evaluation_natural.jsonl \
  --pdf narrative \
  --output current-route-gate \
  --misses
```

Table/figure evaluation:

```bash
uv run python -m eval.run_eval \
  --input eval/pdf/evaluation_structured.jsonl \
  --pdf structured \
  --output current-route-gate \
  --misses
```

Excel evaluation:

```bash
uv run python -m eval.run_eval \
  --input eval/excel/evaluation_excel.jsonl \
  --excel \
  --output current-route-gate
```

Results are written to `eval/pdf/results/narrative/`,
`eval/pdf/results/structured/`, or `eval/excel/results/` as
`YYYY-MM-DD-FEATURE.json`. Add `--oracle` to evaluate only the suite's own lane.
Historical pre-unification results remain in the PDF result subfolders, but
should not be treated as measurements of the new unified schema.

## Tests

The unit tests do not connect to AWS or call external model APIs:

```bash
HF_HUB_OFFLINE=1 uv run --frozen python -m unittest discover -s tests
```

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
