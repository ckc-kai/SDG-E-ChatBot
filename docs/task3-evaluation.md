# Task 3 evaluation plan

## What is being measured

Gold-evidence evaluation supplies the benchmark evidence directly to Task 3:

```text
benchmark question + gold evidence
  -> prompt
  -> answer model
  -> structured-output parsing
  -> citation validation
```

It measures Task 3 generation behavior separately from Task 2 retrieval. It
must not be reported as end-to-end RAG performance.

The fixed smoke suite is `eval/task3_gold_smoke_suite.json`. Its ten cases cover
simple facts, numeric changes, multi-part answers, list completeness, risk
methodology, PSPS content, errata, and both two-chunk benchmark questions.

## Automated checks

For each case, record:

1. Whether Task 3 returned a normal response or `answer_generation_failed`.
2. Whether the model returned valid structured JSON.
3. Citation precision: the fraction of selected IDs that are gold IDs.
4. Citation recall: the fraction of gold IDs selected by the model.
5. Whether citation metadata was copied from the input chunks.
6. Input tokens, output tokens, total tokens, and Bedrock latency.

Exact string match is diagnostic only. A grounded paraphrase can be correct
without matching the benchmark excerpt word for word.

## Human review rubric

Score each dimension from 0 to 2:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Correctness | Incorrect or contradicted | Partly correct | Fully correct |
| Completeness | Misses the answer | Misses one requested part | Answers every requested part |
| Groundedness | Adds unsupported claims | Mostly grounded with a minor unsupported detail | Every material claim is supported |
| Citation support | Citation does not support the answer | Citation supports only part | Selected citation(s) support the full answer |
| Concision | Distracting or excessive | Understandable but wordy | Direct and appropriately brief |

A smoke case passes when:

- no generation error occurs;
- correctness, groundedness, and citation support each score 2;
- completeness scores at least 1; and
- every returned citation ID passes programmatic validation.

Any score of 0 requires reviewing the raw model output, prompt, and evidence
before changing the prompt.

## Commands

Inspect the full suite without calling AWS:

```powershell
python scripts/evaluate_task3_gold.py `
  --mode dry-run `
  --suite eval/task3_gold_smoke_suite.json `
  --limit 10
```

After Bedrock approval, run one case first:

```powershell
python scripts/evaluate_task3_gold.py `
  --mode bedrock `
  --suite eval/task3_gold_smoke_suite.json `
  --limit 1
```

Only after reviewing the first response, increase `--limit` to 5 and then 10.

## Insufficient-context evaluation

The answerable benchmark suite does not measure insufficient-context accuracy.
That behavior is covered by deterministic unit tests for now. A reviewed
negative dataset should be added later from real unanswerable user questions;
synthetically hiding known evidence can be useful for engineering tests but
should not be reported as real-world retrieval performance.
