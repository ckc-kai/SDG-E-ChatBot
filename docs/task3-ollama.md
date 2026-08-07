# Task 3 local Ollama fallback

## Purpose

Ollama lets Task 3 use a real local language model while Amazon Bedrock access
is pending. It replaces only the answer model. Task 2 retrieval, Task 3 prompt
construction, response parsing, citation validation, and the Task 4 contract
stay unchanged.

```text
question + Task 2 chunks
  -> Task 3 grounded prompt
  -> local Ollama /api/chat
  -> schema-constrained model JSON
  -> Task 3 parsing and citation validation
  -> public answer response
```

The provider sends a JSON Schema to Ollama so the model is constrained to:

```json
{
  "answer": "grounded answer",
  "cited_chunk_ids": ["chunk-id"],
  "insufficient_context": false
}
```

Schema-constrained output improves format reliability. It does not guarantee
that an answer is factually correct. Correctness and groundedness still require
benchmark and human review.

The request explicitly disables model thinking output. Task 3 needs the final
schema-constrained answer, and allowing a small reasoning model to spend the
entire output budget on hidden thinking can leave `message.content` empty.

## One-time local setup

Install Ollama separately from the Python project, then pull a model suitable
for the available RAM or GPU. For example:

```powershell
ollama pull qwen3:4b
ollama list
```

Ollama normally runs in the background on Windows. Verify the local API:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

No AWS credentials or API key are needed for a local Ollama endpoint.

## Configuration

Set non-secret environment values in the shell or a local ignored `.env` file:

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_MAX_TOKENS=500
OLLAMA_TEMPERATURE=0
OLLAMA_TIMEOUT_SECONDS=120
OLLAMA_CONTEXT_TOKENS=4096
OLLAMA_TOKEN_SAFETY_FACTOR=1.25
```

`temperature=0` is used for more repeatable evaluation. Larger models may
produce better answers but require more memory and run more slowly.

Task 3 protects the model context independently of Task 2's retrieval depth.
It does not impose a fixed Top-K limit. The prompt budget is the configured
context window minus the maximum output tokens. Ranked chunks are added while
they fit that budget, so Top 10 may all be used when they are short enough. An
unexpectedly oversized Top-1 chunk is truncated only if it cannot fit by
itself; only the prompt copy is shortened, while the original chunk and its
citation metadata are not changed.

`OLLAMA_TOKEN_SAFETY_FACTOR` accounts for tokenizer mismatch. Task 2's stored
`token_count` is produced by the embedding tokenizer, while the answer model
may count the same text differently. The default `1.25` charges estimated
prompt tokens at 125% of their measured value. This value is based on local
Qwen measurements where actual input was about 1.22--1.23 times the stored
estimate. It remains a token budget, not a fixed Top-K limit. Evaluation logs
should be used to recalibrate it for each answer-model family.

Task 4 can use the shared provider factory without containing Ollama-specific
logic:

```python
from generation import AnswerService, create_provider_from_env

service = AnswerService(create_provider_from_env())
```

Changing `TASK3_PROVIDER` to `bedrock` later switches the model provider while
leaving the request, prompt, validation, and public response path unchanged.

## Test Task 3 with benchmark gold evidence

Start with one case:

```powershell
python scripts/evaluate_task3_gold.py `
  --mode ollama `
  --suite eval/task3_gold_smoke_suite.json `
  --limit 1
```

After reviewing the answer and citations, increase `--limit` to 5 and then 10.
This isolates Task 3 because it supplies benchmark gold evidence directly.
The JSON report includes each question, input chunks, exact prompt, raw Ollama
output, validated public response, usage, and automated citation scores.

## Test the local end-to-end retrieval and answer path

When the existing PostgreSQL/pgvector smoke database is running and populated:

```powershell
python scripts/smoke_task2_task3.py --provider ollama
```

This runs one real Task 2 retrieval, adapts its Top-5 results, sends them to the
local model, parses the model response, validates cited chunk IDs, and hydrates
citation metadata. It does not test Task 4 because the FastAPI/React layer is
not yet present in this branch.

## Provider code

- `generation/providers/ollama.py`: local HTTP request and response handling.
- `generation/service.py`: common orchestration shared with mock and Bedrock.
- `generation/prompting.py`: compact grounded prompt shared by all providers.
- `generation/citation_validation.py`: rejects unknown IDs and copies trusted
  metadata from Task 2 chunks.

The implementation uses Python's standard library HTTP client, so no new
Python package or lockfile change is required.
