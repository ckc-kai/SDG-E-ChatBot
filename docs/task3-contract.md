# Task 3 and Task 4 contract

## What Task 4 gives Task 3

The expected flow is for Task 4 to call Task 2 retrieval and then give Task 3:

1. `request_id`: the ID for this user request.
2. `question`: the user's original question.
3. `ranked_results`: the reranked chunks returned by Task 2.

The runtime input is Task 2's real `RankedResult` objects. Files under
`eval/pdf/results/` are evaluation reports rather than runtime chunks.

Task 3 already supports the current Task 2 structure:

```text
RankedResult
  query_object
    chunk_id
    source_pdf
    content
    page_start
    page_end
    breadcrumb
    ...
  rerank_score
```

`distance` and `rerank_score` are optional. Task 4 does not need to add them.

## How Task 4 calls Task 3

Task 3 currently provides a Python API, not an HTTP endpoint.

```python
from generation import AnswerRequest, adapt_ranked_results

# 1. Call Task 2.
ranked_results = retrieval.query.retrieve(question, conn)

# 2. Convert Task 2 results to Task 3 chunks.
chunks = adapt_ranked_results(ranked_results)

# 3. Call Task 3.
request = AnswerRequest(
    request_id=request_id,
    question=question,
    chunks=chunks,
)
response = answer_service.answer(request)

# 4. Return the minimal public result through FastAPI.
return response.to_public_dict()
```

Task 4 can wrap this call in its FastAPI endpoint. Task 3 handles prompt
construction, model calls, structured-output parsing, insufficient-context
handling, and citation checks.

## What Task 3 returns

Task 3 gives Task 4 one of the public results below.

### Supported answer

```json
{
  "request_id": "req_001",
  "answer": "The projected target is 100%.",
  "cited_chunk_ids": ["601"],
  "citations": [
    {
      "chunk_id": "601",
      "source_pdf": "WMP.pdf",
      "page_start": 375,
      "page_end": 376,
      "breadcrumb": "8 Wildfire Mitigations > ..."
    }
  ],
  "insufficient_context": false
}
```

Task 4 can return this object to the frontend. The `insufficient_context` flag
lets the frontend distinguish an insufficient-evidence response from a supported
answer.

### Insufficient context

If Task 2 returns no chunks, Task 3 does not call the model:

```json
{
  "request_id": "req_002",
  "answer": "The provided evidence is insufficient to answer the question.",
  "cited_chunk_ids": [],
  "citations": [],
  "insufficient_context": true
}
```

If chunks are present but do not contain enough evidence, the answer may briefly
describe what is missing. Its wording may vary, so Task 4 should check
`insufficient_context` instead of matching the answer text:

```json
{
  "request_id": "req_003",
  "answer": "The evidence does not provide the projected 2025 target.",
  "cited_chunk_ids": [],
  "citations": [],
  "insufficient_context": true
}
```

### Answer-generation failure

A model timeout, connection failure, invalid model JSON, or a supported answer
without any valid citation produces the same minimal public error:

```json
{
  "request_id": "req_004",
  "error": "answer_generation_failed"
}
```

Task 3 logs the specific cause internally. The public result does not expose
provider details, exception messages, credentials, or stack traces.

For example, if the model claims that its answer is supported but cites only an
unknown chunk ID, Task 3 rejects the answer and returns the error above. Task 4
does not need to inspect or validate model-generated IDs itself.

### How Task 4 distinguishes the results

- `error` is present: Task 3/model processing failed.
- No `error` and `insufficient_context=true`: Task 3 worked, but the retrieved
  evidence was not enough to answer.
- No `error` and `insufficient_context=false`: Task 3 returned a supported
  answer with validated citations.

`insufficient_context=true` does not always mean Task 2 has a bug. The answer
may also be missing from the current documents or lost during parsing.

## Integration notes

1. Please keep the same `request_id` from request to response.
2. Please preserve Task 2 `chunk_id` values so citation validation can match the
   original chunks.
3. The returned `citations` have already been validated, so Task 4 can use them
   directly instead of rebuilding citation metadata from model text.
4. `page_start/page_end` are currently zero-based PDF page indexes, and
   `page_end` is exclusive. Task 4 can convert these values for display; for
   example, `page_start + 1` gives the first physical PDF page number.

## Error behavior

An empty retrieval result is an insufficient-context response, not a system
error. Model timeouts, model connection failures, and invalid model output are
answer-generation failures. A claimed supported answer with no valid citation is
also an answer-generation failure. Task 3 defines the JSON above but does not
choose the webpage message or HTTP status; Task 4 can decide that mapping.

This keeps the first integration small. We can add optional fields or streaming
later if the application needs them.
