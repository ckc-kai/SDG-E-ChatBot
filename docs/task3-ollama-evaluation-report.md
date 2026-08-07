# Task 3 Ollama evaluation report

Date: 2026-08-06
Branch: `dn_dev`
Provider: local Ollama 0.32.6
Model: `qwen3:0.6b` (`Q4_K_M`, 751.63M parameters)
Evaluation file: `eval/task3-results/2026-08-06-qwen3-0.6b-gold-suite-10.json`

## Scope and cost

This was a real local-model test, not a scripted mock. All inference requests
went to `http://127.0.0.1:11434/api/chat`. Ollama reported that the model ran
100% on the local NVIDIA GPU. No AWS, Bedrock, Anthropic, Ollama Cloud, or other
paid inference API was called, so the cloud/API cost of this run was $0.

The suite supplies benchmark gold evidence directly to Task 3. It tests prompt
construction, local model generation, structured JSON, citation selection,
validation, and response hydration. It does not test Task 2 retrieval accuracy.

## Call path

```text
benchmark question + gold chunks
  -> AnswerRequest
  -> build_prompt()
       question
       allowed citation IDs
       breadcrumb + chunk text
  -> OllamaProvider.generate()
  -> local POST /api/chat
       model=qwen3:0.6b
       stream=false
       think=false
       temperature=0
       JSON response schema
  <- raw model JSON
  -> parse_model_answer()
  -> validate_and_hydrate_citations()
       reject unknown IDs
       copy source/page metadata from trusted input chunks
  -> AnswerResponse.to_public_dict()
  -> Task 4
```

The model decides only the answer, cited chunk IDs, and insufficient-context
flag. It never supplies source filenames or page metadata.

## Automated results

| Measure | Result |
| --- | ---: |
| Cases | 10 |
| Provider/parsing errors | 0 |
| Post-validation citation precision | 100% |
| Post-validation citation recall | 100% |
| Cases with an invalid raw citation | 0/10 |
| Cases marked insufficient | 0/10 |
| Input tokens | 6,495 |
| Output tokens | 975 |
| Total tokens | 7,470 |
| Mean local latency | 1,058 ms |
| Median local latency | 766 ms |
| Latency range | 554-3,454 ms |
| Paid API cost | $0 |

Citation precision and recall measure whether the returned IDs are members of
the benchmark's gold chunk set. They do not measure whether the answer text is
factually correct or complete. The 100% citation scores therefore show that
all returned IDs were accepted gold IDs in this run, not that all ten answers
were correct.

The benchmark's `expected_answer` values are long source excerpts rather than
concise reference answers, so exact string match was 0/10 and is not a useful
answer-quality metric for this run.

Some benchmark evidence also contains upstream extraction/encoding artifacts
such as replacement glyphs in punctuation. Task 3 intentionally preserves the
retrieved text, so cleaning those artifacts belongs in Task 1 or in an agreed
adapter pre-processing step rather than in the model response validator.

## Manual answer review

| ID | Input chunk IDs | Raw model answer summary | Review |
| --- | --- | --- | --- |
| `wmp_eval_0001` | `1` | `1` | Fail: evidence says `25%` |
| `wmp_eval_0002` | `4` | `200 to 267`; explained the changed approach | Pass |
| `wmp_eval_0005` | `17` | Listed all four SMS processes | Pass |
| `wmp_eval_0042` | `256,257` | Summarized situational-awareness tools and PSPS updates | Pass |
| `wmp_eval_0050` | `325` | Product of LoRE and CoRE | Pass |
| `wmp_eval_0083` | `584` | Wind, air dryness, vegetation dryness, grass greenness | Pass |
| `wmp_eval_0114` | `859,860` | Listed all six capital-cost categories | Pass |
| `wmp_eval_0135` | `1109` | Returned only two water providers and incomplete titles | Fail |
| `wmp_eval_0140` | `1150` | Returned only English and Spanish | Fail: omitted the additional prevalent languages in the evidence |
| `wmp_eval_0149` | `1208` | Incorrect numbering in Table 9.2 column 1 | Pass |

Manual result: 7 pass and 3 fail. This is acceptable as an
integration smoke test for a sub-1B local model, but it is not strong enough to
replace the planned Bedrock model for final answer-quality evaluation. The
main weakness is completeness on multi-part or long-list questions.

A short prompt experiment added explicit instructions to copy every exact
number/name and list every requested item. On this 0.6B model it caused two
invalid responses in ten cases rather than improving the long-list answers, so
the change was reverted. The final artifact above is from the restored compact
prompt and has zero provider/parsing errors. This is another reason to compare
prompt versions with a fixed suite instead of assuming a longer instruction is
better.

## Representative raw and validated result

Question:

```text
What percentage change in initiative risk requires SDG&E to submit a change
order report under Energy Safety's 2023-2025 WMP guidelines?
```

Input evidence:

```text
chunk_id: 1
source: 2023-12-19_SDGE_2023_Change Order Report_R1.pdf
page_start: 2
page_end: 3
text excerpt: ...increase or decrease of 25%...
```

Raw Ollama output from the final run:

```json
{
  "answer": "1",
  "cited_chunk_ids": ["1"],
  "insufficient_context": false
}
```

Validated public response:

```json
{
  "request_id": "wmp_eval_0001",
  "answer": "1",
  "cited_chunk_ids": ["1"],
  "citations": [
    {
      "chunk_id": "1",
      "source_pdf": "2023-12-19_SDGE_2023_Change Order Report_R1.pdf",
      "page_start": 2,
      "page_end": 3,
      "breadcrumb": "Change Order"
    }
  ],
  "insufficient_context": false
}
```

The response is structurally valid and its citation metadata is trustworthy,
but its answer is wrong. This demonstrates why schema/citation validation and
answer-quality evaluation must be reported separately. The full JSON result
contains every question, full input chunk, exact prompt, raw model JSON,
validated response, usage, and automated scores.

## Real Task 2 to Task 3 smoke test

A separate test exercised the current local runtime path instead of supplying
gold chunks directly:

```text
question
  -> Task 2 retrieve(mode=raw, top_k=5, rerank_top_k=5)
  -> five RankedResult objects
  -> adapt_ranked_results()
  -> Task 3 prompt
  -> local Ollama
  -> citation validation and metadata hydration
  -> public response
```

The local pgvector database contains ten chunks from one real PDF, not the
full corpus. For the question below, Task 2 returned chunk IDs `3`, `4`, `6`,
`10`, and `7`, with rerank scores `7.7123`, `6.9026`, `3.8329`, `3.8068`, and
`3.1706`, respectively. Higher rerank score is more relevant in this result.

Question:

```text
What is the target increase for SDG&E's Strategic Pole Replacement program in
2024, and why was it changed?
```

The two leading evidence chunks contained:

```text
chunk 3: The proposed changes to Strategic Pole Replacement in 2024 are due
to a change in approach...

chunk 4: ...increase the scope of Strategic Pole Replacement... The target
increase for this initiative is 200 to 267.
```

Raw local-model output:

```json
{
  "answer": "The target increase for SDG&E's Strategic Pole Replacement program in 2024 is 267 poles, as per the evidence. The change was due to a shift in approach resulting in a higher target increase from 200 to 267.",
  "cited_chunk_ids": ["3", "6"],
  "insufficient_context": false
}
```

The answer is grounded and substantially correct. However, chunk `6` concerns
Distribution Communications rather than Strategic Pole Replacement. Current
validation can prove that `6` was present in this request and safely hydrate
its metadata, but it cannot prove semantic support. This is a known quality
limitation of the 0.6B model and should become a separate citation-support
evaluation or stricter validation step.

The complete runtime artifact, including all five chunks, exact prompt, raw
model output, and public response, is stored in
`eval/task3-results/2026-08-06-qwen3-0.6b-task2-task3-smoke.json`.

## Recommended Task 4 integration

Task 4 should call Task 3 on the backend. React should not call the local
Ollama port directly.

```python
from generation import (
    AnswerRequest,
    AnswerService,
    adapt_ranked_results,
    create_provider_from_env,
)

answer_service = AnswerService(create_provider_from_env())


def answer_from_retrieval(request_id, question, ranked_results):
    request = AnswerRequest(
        request_id=request_id,
        question=question,
        chunks=adapt_ranked_results(ranked_results),
    )
    return answer_service.answer(request).to_public_dict()
```

Local configuration:

```text
TASK3_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:0.6b
OLLAMA_TEMPERATURE=0
OLLAMA_MAX_TOKENS=500
OLLAMA_TIMEOUT_SECONDS=300
```

If FastAPI runs in a container, `127.0.0.1` refers to that container rather
than the Windows host. Task 4 must either run the backend on the host or agree
on a secured host/container Ollama endpoint. The endpoint should not be exposed
directly to the public internet because the local API has no authentication.

Task 4 remains responsible for choosing the HTTP status for a Task 3
`ErrorResponse`. Task 3's stable public error body remains:

```json
{"request_id": "req_001", "error": "answer_generation_failed"}
```

## Next evaluation steps

1. Add reviewed concise reference answers; do not use source excerpts for exact
   string matching.
2. Add semantic correctness/completeness scoring plus the existing human rubric.
3. Finish downloading and test a 4B or 8B local model for multi-part questions.
4. Run the same ten cases through Bedrock after access is approved.
5. Repeat Task 2-to-Task 3 testing after the full corpus is ingested and the
   current Task 2 branch is integrated.
