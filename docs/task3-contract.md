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

Task 3 gives Task 4 this public result:

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

An empty retrieval result is not a system error. Task 3 returns a normal result
with `insufficient_context=true` and no citations.

If model generation times out, the model connection fails, or the model returns
invalid structured output, Task 3 records the detailed exception in its internal
log and gives Task 4 this result:

```json
{
  "request_id": "req_001",
  "error": "answer_generation_failed"
}
```

Task 3 reports the failure without choosing the webpage message or HTTP status.
Task 4 can decide that mapping as part of the API and frontend behavior. For
example, a provider connection failure could be mapped to `502 Bad Gateway`.

Raw model/provider error details stay in Task 3 internal logs and are not part
of the public error JSON.

This keeps the first integration small. We can add optional fields or streaming
later if the application needs them.
