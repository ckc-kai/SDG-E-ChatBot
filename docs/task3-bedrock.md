# Task 3 Amazon Bedrock development

## Current status

The Task 3 Bedrock integration uses the Amazon Bedrock Converse API. Account
allowlisting is still required before a live call can succeed. Unit tests and
gold-evidence dry runs do not call AWS and do not cost anything.

## Code structure

```text
Task 4 / evaluation script
  -> AnswerRequest
  -> AnswerService
       -> build_prompt()
       -> BedrockProvider.generate()
            -> bedrock-runtime Converse API
            <- raw model text + usage
       -> parse_model_answer()
       -> validate_and_hydrate_citations()
  <- AnswerResponse or ErrorResponse
```

- `generation/providers/bedrock.py`: Bedrock-specific request, response, usage,
  and configuration handling.
- `generation/service.py`: provider-independent answer orchestration.
- `generation/prompting.py`: sends only the question, chunk IDs, optional
  breadcrumbs, and chunk text to the model.
- `generation/citation_validation.py`: accepts only IDs from the input request
  and copies trusted citation metadata from those chunks.
- `scripts/evaluate_task3_gold.py`: isolates Task 3 by using benchmark gold
  evidence rather than Task 2 retrieval output.

## Objects

`BedrockProvider` implements the existing `ModelProvider` interface:

```python
raw_text = provider.generate(prompt)
```

It accepts an injected client. Production can pass a boto3 `bedrock-runtime`
client; tests pass a fake client and never use AWS.

`BedrockUsage` stores the latest Converse metadata:

```text
input_tokens
output_tokens
total_tokens
latency_ms
```

Usage remains internal and is not added to the Task 4 public response.

## Configuration

Required for a live call:

```text
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=<exact model or inference-profile ID>
BEDROCK_MAX_TOKENS=500
BEDROCK_TEMPERATURE=0
```

Do not put AWS access keys, secret keys, session tokens, or account IDs in the
repository. boto3 should use its normal AWS credential chain.

`boto3` is declared in `pyproject.toml` and locked in `uv.lock`. A normal
project environment sync will install it; no AWS request occurs during install.

## Network-free checks

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

Inspect five real benchmark prompts without calling a model:

```powershell
python scripts/evaluate_task3_gold.py --mode dry-run --limit 5
```

Inspect selected questions:

```powershell
python scripts/evaluate_task3_gold.py --mode dry-run `
  --id wmp_eval_0001 `
  --id wmp_eval_0002
```

Inspect the fixed ten-case smoke suite:

```powershell
python scripts/evaluate_task3_gold.py --mode dry-run `
  --suite eval/task3_gold_smoke_suite.json `
  --limit 10
```

See `docs/task3-evaluation.md` for case selection and the human review rubric.

## Live check after approval

After account access, credentials, boto3, and the exact model ID are ready:

```powershell
python scripts/evaluate_task3_gold.py --mode bedrock --limit 1
```

Then increase the limit to 5 or 10 only after reviewing the first response.

The gold-evidence result measures Task 3 generation and citation behavior. It is
not an end-to-end retrieval score because Task 2 retrieval is bypassed.
