# Latest Task 1 -> Task 2 -> Task 3 live evaluation

Date: 2026-08-06

## Scope and reproducibility

This evaluation used a real local pipeline rather than benchmark gold chunks:

```text
source PDF
  -> latest Task 1 document parsing
  -> latest Task 2 chunk preparation, embedding, and PostgreSQL/pgvector storage
  -> latest Task 2 hybrid chunk retrieval and reranking
  -> current Task 3 adapter and token-budget prompt
  -> local Ollama model
  -> Task 3 parsing and citation validation
```

- Task 1/2 code: `origin/ckc_dev` commit
  `3c2bfb16ec65da85a44f8ca46de18e8c4e428c20`.
- Task 3 code: current uncommitted `dn_dev` working tree.
- Database: a new isolated `sdge_task123_eval10` database in the existing
  local pgvector container.
- Source: `2023-12-19_SDGE_2023_Change Order Report_R1.pdf`.
- Ingest result: one document and 11 narrative chunks; no table or figure
  chunks.
- Embedding model: `BAAI/bge-base-en-v1.5`, raw plus contextual embeddings.
- Retrieval: hybrid union, query rewrite off, retrieval pool 30, rerank Top 10.
- Reranker used: cached
  `cross-encoder/ms-marco-MiniLM-L6-v2`. The repository's latest example
  config names `BAAI/bge-reranker-base`, but its download previously stalled,
  so this run does not compare that default reranker.
- Models: `qwen3:0.6b`, `qwen3:4b`, and `qwen3:8b` through local Ollama.
- Ollama settings: context 4096, maximum output 500, temperature 0, thinking
  disabled, JSON Schema output.
- AWS/Bedrock was not called and this evaluation incurred no API cost.

Only one source PDF is locally available. Three questions are existing
repository benchmark questions (`wmp_eval_0001` through `0003`). Seven were
written before model execution from facts in the newly parsed chunks. The
seven additional questions are useful live-pipeline tests, but they are not an
independently reviewed benchmark and must not be presented as such.

## Document parsing and chunk-ingest observations

Task 1's narrative document parsing completed successfully. The Task 2 ingest
path then produced the same 11 narrative chunks as the earlier isolated run.
Important observations:

1. Some stored Task 2 chunks contain more than one program. Chunk 5 contains the end of
   Strategic Pole Replacement and most of Distribution Infrared Inspections,
   while chunk 6 contains the end of Infrared Inspections, Wireless Fault
   Indicators, and Distribution Communications Reliability Improvements.
2. Their breadcrumb is nevertheless labeled as Distribution Communications
   Reliability Improvements. This is misleading metadata for facts about the
   other programs and can weaken contextual embeddings and displayed citation
   context.
3. A few Task 2 chunk boundaries split text mid-word or mid-sentence: chunk 5 ends with
   `does not anti`, chunk 7 begins with `MP.549`, and chunk 7 ends before the
   sentence completes. Retrieval still found the needed facts in this small
   corpus, but these are real chunk-preparation issues in the current Task 2
   ingest/retrieval implementation.
4. Ingest printed a `518 > 512` tokenizer warning while measuring contextual
   embedding input. The current code subsequently removes supplementary
   context and completed both raw and contextual embedding successfully, so
   this run did not fail. The warning path should still be cleaned up and
   verified with a test proving that the actual encoded contextual input never
   exceeds the embedding model limit.
5. Structured extraction was disabled for this local run. The source is a
   short narrative change-order report, so this evaluation says nothing about
   the latest Docling table/figure path or Excel ingestion.

## Retrieval results

The expected evidence was retrieved within Top 10 for all ten questions:

| Question | Best expected-evidence rank | Observation |
|---|---:|---|
| WMP 0001, 25% threshold | 1 | Correct |
| WMP 0002, pole target/reason | 1 | Second supporting chunk ranked 2 |
| WMP 0003, weather stations/purpose | 1 | Correct |
| Live 0004, 2023 changes | 2 | Fixed Backup Power incorrectly ranked 1 |
| Live 0005, pole expansion/postponement | 1 | Correct |
| Live 0006, infrared target/reason | 1 | Second useful continuation ranked 3 |
| Live 0007, fault indicators | 1 | Correct |
| Live 0008, communications target | 1 | Related continuation chunks ranked 2/3 |
| Live 0009, backup power | 1 | Correct |
| Live 0010, inspection targets | 1 | Correct |

Aggregate retrieval on this small, locally authored set:

- Hit@1: 9/10.
- Hit@3: 10/10.
- Recall@10: 10/10.

The single Top-1 miss is meaningful. For “What changes does SDG&E request for
its 2023 mitigation initiatives?”, the reranker gave Fixed Backup Power chunk
8 a score of 8.2413 and the exact 2023 “no changes” chunk 3 a score of 8.1582.
The 0.6B model followed the incorrect Top-1 topic even though the correct chunk
was also present at rank 2. The 4B and 8B models selected the correct evidence.

These values cannot be treated as general retrieval accuracy: the database has
only 11 chunks from one PDF, and seven questions were authored directly from
those chunks.

## Evaluation method

Before model execution, the ten questions were assigned 36 required facts,
including requested numbers, before/after targets, and reasons. A literal
required-fact recall is reported as a reproducible diagnostic. It is not a
complete semantic quality score: for example, the 8B answer says “alternative
mitigation approaches,” while one literal alternative expected “additional
mitigation approaches,” producing a false negative even though the meaning is
supported.

Answers were therefore also manually reviewed as pass, partial, or fail:

- Pass: all requested parts are correctly answered from retrieved evidence.
- Partial: core answer is correct but one requested reason or component is
  omitted.
- Fail: materially wrong answer or failure to answer the question.

## Model comparison

| Model | Literal facts | Valid responses | Manual pass | Partial | Fail | Mean model latency | Mean cited IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 28/36 (77.8%) | 10/10 | 6 | 3 | 1 | 2.06 s | 6.9 |
| Qwen3 4B | 33/36 (91.7%) | 10/10 | 9 | 1 | 0 | 4.92 s | 1.0 |
| Qwen3 8B | 32/36 (88.9%) | 10/10 | 9 | 1 | 0 | 13.39 s | 1.3 |

The 8B literal result is understated by at least one wording-equivalence false
negative. Another required fact—the old value of 300 for Wireless Fault
Indicators—is useful context but was not strictly necessary to answer the
question asking for the revised target and reason. Manual review is therefore
more informative than declaring 4B more accurate from 33 versus 32 alone.

### Qwen3 0.6B

Strengths:

- Returned valid schema-constrained JSON for all ten questions.
- Correctly answered straightforward numerical questions such as 25%, 222
  weather stations, 60 to 15 base stations, and 15,450 to 16,394 inspections.

Problems:

- Failed the 2023-initiatives question. It answered with unrelated 2024 Fixed
  Backup Power and communications changes because the misleading chunk was
  ranked first.
- Partially answered the pole replacement reason, infrared-inspection reason,
  and Fixed Backup Power focus question.
- Cited 69 IDs across ten responses, averaging 6.9 per answer. Most answers
  cited many legal-but-irrelevant IDs. Current citation validation confirms
  only that an ID was supplied to the model; it does not prove semantic
  support.

Conclusion: 0.6B is suitable for connection and response-format smoke tests,
but not for the answer-quality baseline.

### Qwen3 4B

Strengths:

- Nine complete answers and no failures.
- Each response cited exactly one directly supporting chunk.
- Correctly ignored the misleading Top-1 result for the 2023 question and used
  rank-2 chunk 3.
- Best speed/quality balance in this local test.

Problem:

- For the two-part weather-station question, it returned only `222` and
  omitted the requested purpose of monitoring dangerous fire weather. The
  correct evidence was Top 1, so this is Task 3/model completeness rather than
  a retrieval miss.

### Qwen3 8B

Strengths:

- Nine complete answers and no failures.
- Correctly answered both parts of the weather-station question that 4B missed.
- Citations were concise and semantically relevant: 13 IDs across ten answers.
- Correctly ignored the misleading Top-1 result for the 2023 question.

Problem:

- For Strategic Pole Replacement, it correctly stated 200 to 267 and the
  expanded scope, but omitted the more specific fact that approximately 250
  HFTD poles required pole-loading remediation. This was therefore marked
  partial.
- It was about 2.7 times slower than 4B. Ollama reported the 6.0 GB model as
  approximately 70% GPU / 30% CPU on the 6 GB RTX 3060 Laptop GPU, explaining
  part of the latency.

Conclusion: this ten-question set does not demonstrate that 8B is consistently
more accurate than 4B. It changes which detail is missed and is materially
slower. A larger, independently reviewed benchmark is required before choosing
between them.

## Prompt/context observations

Task 2 returned ten results and all ten fit Task 3's current estimated prompt
budget, so every model received all ten for every question. Real Ollama input
counts were approximately 3,367 to 3,867 tokens within a 4,096-token context.
The largest generated answer used 124 tokens, so none of these calls was
truncated.

However, Task 3 estimated that the prompt fit within a 3,596-token budget
(4,096 context minus a 500-token output reserve), while Ollama later reported
up to 3,867 actual input tokens. The reason is that Task 2's `token_count` uses
the BGE embedding tokenizer, while Qwen uses a different tokenizer. This is a
real Task 3 safety issue: token budgeting by the embedding tokenizer is not an
exact guarantee for the answer model.

There should still be no fixed Top-5 rule. Recommended fixes are, in order:

1. Add an answer-provider token-count interface when the provider exposes one.
2. Otherwise apply a configurable safety margin to estimated prompt tokens.
3. Log estimated versus provider-reported input tokens during evaluation and
   calibrate the margin per model family.
4. Keep retrieval depth and prompt selection separate. Task 2 may retrieve ten
   candidates, while Task 3 includes as many as safely fit.

### Task 3 fix applied after this comparison

Task 3 now applies a configurable tokenizer safety factor. A first value of
1.15 was rejected by a real three-question rerun because Qwen still reported
3,644--3,767 input tokens against the 3,596-token prompt budget. The default
was therefore calibrated to 1.25.

With the same real retrieved chunks and Qwen3 4B, the final 1.25 setting gave:

| Question | Chunks included | Safety-adjusted estimate | Actual Qwen input |
|---|---:|---:|---:|
| WMP 0001 | 9 | 3,497 | 3,415 |
| WMP 0002 | 8 | 3,580 | 3,509 |
| WMP 0003 | 9 | 3,419 | 3,381 |

All three actual inputs were below 3,596, all answers were complete and
correct, and citations remained minimal. Task 3 also logs a warning whenever a
provider later reports actual input above the calculated budget, allowing this
factor to be recalibrated rather than silently overflowing. This is still a
token-budget policy, not a fixed Top-K rule.

Token capacity alone is not a relevance policy. The 0.6B failure shows that ten
chunks can fit technically while still adding enough irrelevant content to
confuse a small model. Any score cutoff or score-gap policy must be calibrated
within each Task 2 result group; rerank scores are not globally comparable
across narrative, table, figure, and Excel groups.

## Problems by component

### Task 1: document parsing

1. Narrative PDF text parsing succeeded for the one local Change Order PDF.
2. Table, figure, OCR, and Excel parsing remain untested in this environment.

Recommended next work:

- Re-ingest one table-heavy WMP PDF with Docling enabled before drawing any
  conclusion about long tables.
- Confirm that Task 1's handoff preserves program headings, section boundaries,
  and table structure for Task 2.

### Task 2: chunk preparation and retrieval

1. Mixed-program chunks, misleading breadcrumbs, and mid-word/mid-sentence
   boundaries are present in the current stored chunks.
2. One exact short answer was rank 2 behind an unrelated 2024 program.
3. Current local run used the MiniLM fallback reranker rather than the latest
   example-config reranker.
4. Scores are not yet calibrated into a reliable relevance cutoff.

Recommended next work:

- Prefer sentence/paragraph-aware chunk boundaries inside the token ceiling.
- Split when a new program heading begins and assign the governing breadcrumb.
- Add regression tests for the observed `does not anti` / `MP.549` boundary.
- Run the repository retrieval benchmark with the intended reranker once its
  model is available, then compare it with MiniLM.
- Evaluate date/year sensitivity and short direct-answer queries.
- Report Hit@1, Recall@5/10, and per-group retrieval metrics on the reviewed
  dataset, not this ten-question local set.
- Decide a calibrated per-group score/score-gap policy before discarding
  lower-ranked evidence.

### Task 3

1. Answer-model token estimation does not exactly match Ollama/Qwen tokens;
   the new 1.25 safety factor mitigates this but is not an exact tokenizer.
2. Small models can cite many irrelevant but valid chunk IDs.
3. Even 4B/8B occasionally omit one part of a multi-part question despite the
   prompt instruction.
4. Current runtime validation checks ID membership and hydrates trusted
   metadata, but does not verify semantic citation support.

Recommended next work:

- Prefer provider-aware token counting when available; retain and monitor the
  measured safety margin otherwise.
- Use 4B as the current local development baseline; retain 0.6B only for fast
  smoke tests.
- Expand the reviewed benchmark before deciding that 8B's latency is worth it.
- Add offline completeness and citation-support metrics. Do not add a paid
  second LLM runtime call until the quality/latency tradeoff is measured.
- Add insufficient-context and adversarial irrelevant-chunk cases; all ten
  questions here were answerable, so refusal quality was not tested.

## Overall conclusion

The one-PDF narrative chain is operational end to end: parsing, dual
embeddings, database ingest, retrieval, reranking, Task 3 structured generation,
and citation hydration all completed without AWS. Retrieval found expected
evidence within Top 3 for every test question.

The clearest current issues are:

1. Task 2 chunk boundaries and breadcrumb accuracy.
2. Task 2 Top-1 reliability and unverified intended reranker configuration.
3. Task 3 semantic citation quality; token budgeting now has a measured safety
   factor but should eventually use provider-aware counting.
4. Model completeness: 0.6B is inadequate; 4B and 8B are both usable but each
   missed one requested detail in this small test.

The full machine-readable artifact contains all ten questions, every retrieved
chunk and score, exact prompts, raw model JSON, validated public responses,
token usage, latency, and required-fact checks:

`eval/task3-results/2026-08-06-latest-task123-three-models-10cases.json`
