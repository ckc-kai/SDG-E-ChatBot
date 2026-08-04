# Unified evaluation runner

Run any evaluation JSONL through one entry point:

```bash
uv run python -m eval.run_eval \
  --input PATH_TO_EVALUATION.jsonl \
  --output FEATURE_NAME
```

The suite is auto-detected. It can also be made explicit with
`--pdf narrative`, `--pdf structured`, or `--excel`. Current real routing
(`narrative structured excel`) is the default; `--oracle` restricts retrieval to
the suite's own lane.

Default outputs:

```text
eval/pdf/results/narrative/YYYY-MM-DD-FEATURE.json
eval/pdf/results/structured/YYYY-MM-DD-FEATURE.json
eval/excel/results/YYYY-MM-DD-FEATURE.json
```

`--output` is a feature label, not a path. The runner always selects the result
directory and prepends `YYYY-MM-DD-`. Existing files are protected; use a
different label or opt into replacement with `--overwrite`. The older
`--feature` spelling remains available as a compatibility alias.

Examples:

```bash
uv run python -m eval.run_eval \
  --input eval/pdf/evaluation_natural.jsonl \
  --pdf narrative \
  --output current-route-gate

uv run python -m eval.run_eval \
  --input eval/pdf/evaluation_structured.jsonl \
  --pdf structured \
  --output current-route-gate

uv run python -m eval.run_eval \
  --input eval/excel/evaluation_excel.jsonl \
  --excel \
  --output current-route-gate
```

The legacy `eval.run_structured_eval` and `eval.run_excel_eval` commands remain
as thin compatibility wrappers.
