# Latest Task 1-3 integration evaluation

Date: 2026-08-06
Task 1/2 commit: `3c2bfb16ec65da85a44f8ca46de18e8c4e428c20`
Task 3: `dn_dev` working tree
Answer model: local Ollama `qwen3:4b`

## Benchmark provenance

The narrative benchmark was introduced by Kaicheng Chu in commit `89fe88d`
(`update ingest and query, create the evaluation dataset and run`) on
2026-07-21. The same commit added 150 rows to both
`eval/pdf/evaluation.jsonl` and `eval/pdf/evaluation_natural.jsonl`, the
evaluation runner, and a persisted retrieval result. Commit `2577e67` updated
the dataset on 2026-07-22.

The progress log says the full PDF corpus was re-ingested, the evaluation data
was audited/repaired, and evaluation was run. Individual rows also contain
notes such as `Full-corpus audit` or `Manually grounded`, plus source PDF, gold
chunk IDs, page ranges, breadcrumbs, and evidence excerpts.

The repository does not contain a reproducible dataset-generation program or
a written annotation protocol showing exactly how the original questions were
authored and reviewed. The defensible description is therefore: a checked-in
full-corpus retrieval benchmark assembled and audited by the Task 2 author,
with persisted gold evidence. It should not be described as an independently
reviewed answer-quality gold standard without additional confirmation.

## Latest repository state

A blob-filtered fetch verified the current remote heads without downloading
the very large historical result blobs:

- `main`: `b859e1b` (merge of current `ckc_dev`)
- `ckc_dev`: `3c2bfb1` (`optimize retrieval`)
- `dn_dev`: `546cad1`
- `sheena_dev`: `fd0c325`

Task 1/2 was run from an isolated detached sparse worktree at `3c2bfb1`.
Nothing was merged into or committed on any branch.

## Test scope and deviations

Only one source PDF is available locally:

```text
2023-12-19_SDGE_2023_Change Order Report_R1.pdf
```

Only three of the 150 narrative benchmark questions use this PDF, so these are
the only cases for which a new live Task 1 -> Task 2 -> Task 3 run can be
claimed. The other 147 cases require their source PDFs or must use persisted
retrieval/gold evidence instead of a live re-ingest.

The latest default configuration uses `BAAI/bge-reranker-base`. Its model
download stalled at zero bytes for ten minutes, so this run used the already
cached and repository-supported
`cross-encoder/ms-marco-MiniLM-L6-v2`. This is a test of the latest retrieval
code and grouped contract, but not a reproduction of the latest default-model
retrieval metrics.

Docling is not installed in the current Python environment. Structured
table/figure extraction was disabled for this run; all three available cases
are narrative cases. The latest code does implement exact table grids in
`structured_data`, but that path has not yet been executed locally.

## Task 1 result

The latest bookmark-aware parser ingested the PDF in 48.7 seconds and produced
11 narrative chunks. The older smoke database contained 10 chunks from the
same PDF. Raw and contextual embeddings were generated and written to a new,
separate `sdge_latest_3c2b` PostgreSQL database.

Ingest emitted this warning:

```text
Token indices sequence length is longer than the specified maximum sequence
length for this model (518 > 512).
```

The configured chunk limit is 400 embedding-model tokens, but contextual
embedding text adds metadata/context. The latest code completed successfully,
but this warning should be checked because truncation or indexing behavior can
affect contextual retrieval quality.

## Task 2 result

The latest grouped dispatcher was called with only the `narrative` evidence
group. Query rewrite was explicitly off, so no Anthropic or AWS request was
made.

| ID | Gold fact | Correct evidence rank | Top rerank score |
| --- | --- | ---: | ---: |
| `wmp_eval_0001` | Change-order threshold is 25% | 1 | 8.5185 |
| `wmp_eval_0002` | Pole target changes 200 to 267 | 1 | 7.4794 |
| `wmp_eval_0003` | 222 weather stations and network purpose | 1 | 6.6808 |

Live retrieval success on this limited sample is therefore Hit@1 = 3/3. This
is encouraging but far too small to estimate full-corpus retrieval accuracy.

The latest public Task 2 contract is now grouped
`EvidenceRetrievalResult`, with independently ranked narrative, table, figure,
and Excel groups. Task 3's current adapter accepts a list of `RankedResult`
objects. This test explicitly selected
`result.groups["narrative"].results`. Task 4 still needs an agreed policy for
which groups to send to Task 3; scores must not be globally sorted across
groups because Task 2 states they are comparable only within a group.

## Task 3 result with ten retrieved chunks

| ID | Prompt input tokens | Result | Review |
| --- | ---: | --- | --- |
| `0001` | 3,748 | `25%`, citation `1` | Pass |
| `0002` | 3,833 | `200 to 267`, reason, citation `4` | Pass |
| `0003` | 3,838 | cited literal `string`; rejected | Error |

The local Ollama context was 4,096 tokens. The third request returned the JSON
schema placeholder rather than a real chunk ID:

```json
{"answer":"string","cited_chunk_ids":["string"],"insufficient_context":false}
```

Task 3 correctly returned `answer_generation_failed` instead of publishing an
unsupported citation. The correct chunk was already retrieval rank 1, so this
was not a Task 2 miss. Context pressure and the small 4B model are the likely
causes.

## Task 3 result with five retrieved chunks

| ID | Prompt input tokens | Result | Review |
| --- | ---: | --- | --- |
| `0001` | 2,155 | `25%`, citation `1` | Pass |
| `0002` | 1,954 | `200 to 267` with reason, citation `4` | Pass |
| `0003` | 2,059 | `222`, citation `9` | Partial |

Top 5 removed the invalid-output failure and reduced prompt input by roughly
45%. The third answer is grounded and correctly cited, but it omits the second
requested part: the network's primary purpose is to monitor dangerous fire
weather conditions and provide situational-awareness/foundational operational
data.

Human answer-quality result for the live Top-5 sample is therefore 2 pass,
1 partial, 0 unsupported answers, and 0 public citation errors.

## Findings by component

### Task 1

- Narrative parsing and stable source/page metadata worked.
- The contextual embedding length warning needs investigation.
- Structured table/figure extraction could not be verified without Docling
  and the corresponding source PDF.

### Task 2

- Correct evidence ranked first for all three available questions.
- The latest default reranker could not be downloaded in this environment, so
  the cached MiniLM reranker was used.
- The grouped output is safer for incomparable content types, but requires a
  Task 4/Task 3 handoff policy.

### Task 3

- Ten chunks nearly exhausted the current model context and caused one invalid
  structured response.
- Five chunks were much more stable and retained the correct evidence.
- Qwen3 4B still missed one part of a two-part question.
- Citation validation prevented the literal `string` ID from reaching Task 4.

## Table benchmark implication

Case `wmp_eval_0135` and chunk `1109` come from the checked-in full-corpus
benchmark, not this new runtime database. Its persisted narrative evidence
spans page indexes 975-982 and flattens multiple table categories, repeated
headers, and page breaks. This is valid evidence of a historical benchmark
chunk-quality problem, but not proof that the latest structured pipeline still
produces the same representation.

The latest Task 1 code already adds Docling table extraction and retains exact
grids in `structured_data`. The right follow-up is to re-ingest the source PDF
with Docling and inspect Table 8-46, not to ask Task 1 to redesign the schema
before checking the current output.

## Recommended next evaluation

1. Finish the Qwen3 8B download and rerun the same live Top-5 cases.
2. Add a prompt-budget policy rather than forwarding a fixed number of chunks
   regardless of token length.
3. Reproduce the latest default BGE reranker after its weights are available.
4. Obtain the full PDF corpus and run all 150 retrieval cases against the clean
   schema.
5. Add concise reviewed reference answers and per-question completeness
   criteria; the current `expected_answer` is often a long source excerpt.
6. Install Docling in an isolated environment and separately evaluate the
   structured table/figure route.

## Artifacts

- `eval/task3-results/2026-08-06-latest-task123-qwen3-4b-3cases.json`
- `eval/task3-results/2026-08-06-latest-task123-qwen3-4b-top5-3cases.json`
