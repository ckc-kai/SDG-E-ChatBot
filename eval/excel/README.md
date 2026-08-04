# Excel evaluation suites

The Excel benchmark is split so release metrics are not distorted by known
future capabilities or by facts that both retrieval corpora contain.

- `evaluation_excel.jsonl` — 80 strict Excel-only questions supported by the
  current deterministic channel. This is the release-gating suite.
- `evaluation_excel_challenge.jsonl` — 24 trends, comparisons, rankings,
  attributes, missing-value, and clarification questions. Reported
  informationally until the live channel supports those shapes.
- `evaluation_excel_cross_corpus.jsonl` — 10 facts expressed as 20 source-cued
  PDF/Excel questions for routing evaluation.
- `manifest.json` — active revision hashes and suite hashes. Evaluation stops
  when a new active workbook makes the gold data stale.

Generate and validate:

```bash
uv run python -m eval.generate_excel_eval
uv run python -m eval.validate_excel_eval
```

Run the release evaluation:

```bash
uv run python -m eval.run_eval \
  --input eval/excel/evaluation_excel.jsonl \
  --excel \
  --output current-route-gate
```

The evaluator reports card retrieval, gold-plan correctness, preferred-lane
selection, and live-channel correctness independently. `expected_answer` is the
gold value; `pdf_evidence` and `provenance` identify acceptable supporting
sources rather than alternate answers.
