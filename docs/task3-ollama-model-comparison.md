# Task 3 Ollama model comparison

Date: 2026-08-06
Branch: `dn_dev`
Models: `qwen3:0.6b` and `qwen3:4b` (`Q4_K_M`)
Provider: local Ollama 0.32.6

## Test controls and cost

Both models received the same ten benchmark questions, gold evidence chunks,
Task 3 prompt, JSON schema, `temperature=0`, `think=false`, output limit, parser,
and citation validator. All inference requests went to the local
`http://127.0.0.1:11434/api/chat` endpoint. No AWS, Bedrock, Anthropic, or
other paid inference API was called, so cloud/API inference cost was $0.

The 4B model was already fully downloaded by the earlier resumable background
pull when this comparison began. Its registered model size is 2.5 GB. The
ten-case 4B benchmark took 26.8 seconds wall-clock, including model startup.
Ollama reported a 3.2 GB loaded model and 100% GPU execution on the local
NVIDIA RTX 3060 Laptop GPU.

## Summary

| Measure | Qwen3 0.6B | Qwen3 4B |
| --- | ---: | ---: |
| Provider/parsing errors | 0/10 | 0/10 |
| Post-validation citation precision | 100% | 100% |
| Post-validation citation recall | 100% | 95% |
| Human review | 7 pass, 3 fail | 8 pass, 2 partial, 0 fail |
| Input tokens | 6,495 | 6,435 |
| Output tokens | 975 | 692 |
| Mean local latency | 1,058 ms | 2,637 ms |
| Median local latency | 766 ms | 1,750 ms |
| Latency range | 554-3,454 ms | 1,246-9,864 ms |
| Cloud/API inference cost | $0 | $0 |

The automated citation metrics do not measure answer correctness. The 4B
recall is 95% because case `wmp_eval_0042` had two gold chunks but the model
cited only `256` and answered only the situational-awareness portion, omitting
the PSPS-model-update evidence in `257`. This case is therefore partial rather
than a fully supported answer.

## Per-question comparison

| ID | Question topic | 0.6B | 4B | 4B review |
| --- | --- | --- | --- | --- |
| `0001` | Change-order risk threshold | `1` | `25%` | Pass; fixes a direct numerical error |
| `0002` | Strategic Pole Replacement target | Correct | Correct, `200` to `267` with reason | Pass |
| `0005` | Four SMS processes | All four | All four, concise | Pass |
| `0042` | Situational awareness and PSPS updates | Covered both, with questionable phrasing | Covered situational awareness only | Partial; omitted PSPS update |
| `0050` | Wildfire/PSPS risk formula | `LoRE x CoRE` | `LoRE x CoRE` | Pass |
| `0083` | SAWTI factors | Correct | Correct and concise | Pass |
| `0114` | Covered-conductor cost components | Correct but very verbose | Six category names only | Pass; substantially more concise |
| `0135` | Water providers and contact titles | Only 2 entries | 11 of 13 entries | Partial; large improvement but still incomplete |
| `0140` | Languages | Unsupported `English and Spanish` | `22 prevalent languages` | Pass; stays within the evidence |
| `0149` | Table 9.2 errata | Correct | Correct | Pass |

The 4B model eliminated the three clear 0.6B failures. It remains incomplete
on a long table and on one two-part question, so model size alone does not
solve table structure, evidence completeness, or citation-support evaluation.

## Exact examples

### Numerical extraction

Question:

```text
What percentage change in initiative risk requires SDG&E to submit a change
order report under Energy Safety's 2023-2025 WMP guidelines?
```

Gold chunk `1` contains `increase or decrease of 25%`.

0.6B raw answer:

```json
{"answer":"1","cited_chunk_ids":["1"],"insufficient_context":false}
```

4B raw answer:

```json
{"answer":"25%","cited_chunk_ids":["1"],"insufficient_context":false}
```

### Long table

Question `wmp_eval_0135` asks for all water providers and designated contact
titles. Gold chunk `1109` contains thirteen entries. The 0.6B model returned
only Summit Estates and Sweetwater. The 4B model returned eleven entries but
omitted Summit Estates and Sweetwater. The improvement is material, but the
answer is still incomplete.

### Evidence-limited language answer

Gold chunk `1150` states that materials are available in the `22 prevalent
languages identified in SDG&E's service territory`; it does not enumerate all
language names. The 0.6B model invented `English and Spanish`, while the 4B
model correctly stayed at `22 prevalent languages`.

## Real Task 2 to Task 3 comparison

Both models received the same current Task 2 Top-5 retrieval for:

```text
What is the target increase for SDG&E's Strategic Pole Replacement program in
2024, and why was it changed?
```

The ranked chunk IDs were `3`, `4`, `6`, `10`, and `7`. Chunk `6` concerns
Distribution Communications and does not directly support the answer.

0.6B output:

```json
{
  "answer": "The target increase ... is 267 poles ... from 200 to 267.",
  "cited_chunk_ids": ["3", "6"],
  "insufficient_context": false
}
```

4B output:

```json
{
  "answer": "The target increase ... is from 200 to 267 poles, due to a change in approach resulting in approximately 250 poles requiring pole loading remediation through other projects.",
  "cited_chunk_ids": ["3"],
  "insufficient_context": false
}
```

The 4B answer is more precise and avoids the unrelated citation. Citation
metadata in the public response was copied from trusted chunk `3`, not supplied
by the model.

## Conclusion

Qwen3 4B is a much more credible local integration and demonstration model
than Qwen3 0.6B. On this small suite it removed all clear failures, improved
numerical grounding and citation selection, and produced more concise output,
at roughly 2.5 times the mean local latency. It is not yet evidence that 4B is
the final production model: the suite is small, uses gold evidence, and still
shows incomplete multi-part and long-table answers.

The next fair comparison should keep this exact suite and add reviewed concise
reference answers plus a completeness rubric. If local hardware permits, an 8B
model can then be compared using the same artifacts before choosing an AWS GPU
instance or Bedrock model.

## Result artifacts

- `eval/task3-results/2026-08-06-qwen3-0.6b-gold-suite-10.json`
- `eval/task3-results/2026-08-06-qwen3-4b-gold-suite-10.json`
- `eval/task3-results/2026-08-06-qwen3-0.6b-task2-task3-smoke.json`
- `eval/task3-results/2026-08-06-qwen3-4b-task2-task3-smoke.json`
