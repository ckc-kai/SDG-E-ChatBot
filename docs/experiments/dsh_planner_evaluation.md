# DSH planner versus original pipeline (27-case A/B, 2026-08-23)

## Experimental design

This report compares two blind runs of all 27 `beta_golden_questions`:

- **DSH-gated:** the deterministic `needs_planning()` gate sent 25 complex
  questions through DSH with local `qwen3:14b`; two simple questions retained
  the original route.
- **Original:** every complex question used the original Ollama planner with
  `qwen3:4b`; simple questions used the original rules and lightweight
  auxiliary judge. DSH was not used.

Every final answer used `deepseek/deepseek-v4-flash`. The original variant had
already been captured for six cases; this run added the remaining 21.

DeepSeek graded each saved result on three independent 0-10 dimensions:

- **R (retrieval):** decomposition, source/modality selection, entity-period
  binding, and retrieved evidence.
- **G (generation):** fidelity and safety given only the evidence actually
  supplied. Missing retrieval evidence is not penalized; a justified partial
  answer or safe refusal can receive 10.
- **P (pipeline):** end-to-end correctness against the frozen gold answer and
  expected behavior.

For a paired comparison, the original R/P grader reused each case's frozen DSH
atomic-requirement checklist. Each P score was capped by the proportion of
requirements supported by a short verbatim excerpt from the final answer. This
removes the earlier source of variance where the two variants could receive
different checklist denominators. G remained a separate gold-blind call.

This is one controlled run on one host, not a statistical model benchmark.

## Result for the newly tested 21 cases

| Metric | DSH-gated | Original | Difference (DSH-original) |
| --- | ---: | ---: | ---: |
| Mean R | 5.857 | 6.000 | -0.143 |
| Mean G | 8.810 | 9.048 | -0.238 |
| Mean P | 2.286 | 2.667 | -0.381 |
| Mean latency | 28.899 s | 14.370 s | +101.1% |
| Median latency | 31.940 s | 15.375 s | +107.7% |
| Total pipeline time | 606.875 s | 301.771 s | +305.104 s |

Across these 21 cases, DSH had a higher P score on **0**, tied on **16**, and
had a lower P score on **5**. The five original-route wins were
`beta_user_002`, `beta_user_008`, `beta_user_014`,
`beta_excel_multi_005`, and `beta_cross_limit_001`.

## Full 27-case result

| Metric | DSH-gated | Original | Difference (DSH-original) |
| --- | ---: | ---: | ---: |
| Mean R | 6.037 | 5.926 | +0.111 |
| Mean G | 8.926 | 9.037 | -0.111 |
| Mean P | 2.519 | 2.667 | -0.148 |
| Mean latency | 29.592 s | 15.475 s | +91.2% |
| Median latency | 31.570 s | 16.010 s | +97.2% |
| Total pipeline time | 798.995 s | 417.827 s | +381.168 s |

DSH had a higher P score on **2** cases, tied on **18**, and had a lower P
score on **7**. Its small mean-R advantage did not translate into a mean-P
advantage. The original route was slightly better overall and substantially
faster in this run.

## Per-case comparison

`P delta` is DSH minus original. The `new 21` rows are the cases added by this
run; `existing six` rows are the earlier A/B subset re-evaluated with the same
shared checklist.

| Case | Run | DSH R/G/P | Original R/G/P | P delta | DSH time | Original time |
| --- | --- | --- | --- | ---: | ---: | ---: |
| beta_user_001 | existing six | 8/8/1 | 8/8/3 | -2 | 44.707 s | 31.381 s |
| beta_user_002 | new 21 | 7/9/0 | 7/8/1 | -1 | 43.487 s | 32.872 s |
| beta_user_003 | new 21 | 8/8/0 | 7/10/0 | 0 | 49.173 s | 35.899 s |
| beta_user_004 | existing six | 3/10/0 | 4/8/0 | 0 | 36.578 s | 17.656 s |
| beta_user_005 | new 21 | 4/10/3 | 6/10/3 | 0 | 23.499 s | 9.414 s |
| beta_user_006 | new 21 | 4/8/0 | 4/8/0 | 0 | 32.167 s | 11.158 s |
| beta_user_007 | new 21 | 6/8/0 | 6/8/0 | 0 | 30.422 s | 15.654 s |
| beta_user_008 | new 21 | 4/8/2 | 4/10/4 | -2 | 32.625 s | 17.129 s |
| beta_user_009 | new 21 | 4/8/0 | 7/8/0 | 0 | 27.716 s | 18.318 s |
| beta_user_010 | new 21 | 3/10/0 | 3/10/0 | 0 | 35.413 s | 16.129 s |
| beta_user_011 | new 21 | 7/10/3 | 4/10/3 | 0 | 35.388 s | 9.069 s |
| beta_user_012 | new 21 | 4/8/0 | 4/8/0 | 0 | 33.965 s | 11.717 s |
| beta_user_013 | new 21 | 5/8/0 | 7/8/0 | 0 | 31.940 s | 15.375 s |
| beta_user_014 | new 21 | 6/8/4 | 4/8/6 | -2 | 34.571 s | 16.992 s |
| beta_user_015 | new 21 | 4/8/1 | 4/8/1 | 0 | 28.188 s | 16.958 s |
| beta_pdf_visual_001 | new 21 | 10/10/8 | 10/10/8 | 0 | 3.972 s | 7.218 s |
| beta_pdf_visual_002 | existing six | 8/10/8 | 2/10/0 | +8 | 26.110 s | 23.656 s |
| beta_pdf_visual_003 | new 21 | 7/10/6 | 7/10/6 | 0 | 20.943 s | 6.119 s |
| beta_pdf_visual_004 | new 21 | 8/10/3 | 8/10/3 | 0 | 2.926 s | 2.971 s |
| beta_pdf_visual_005 | new 21 | 4/10/0 | 4/10/0 | 0 | 8.548 s | 4.882 s |
| beta_excel_multi_001 | existing six | 10/10/8 | 8/10/4 | +4 | 31.570 s | 16.412 s |
| beta_excel_multi_002 | new 21 | 6/7/4 | 8/8/4 | 0 | 43.123 s | 16.010 s |
| beta_excel_multi_003 | new 21 | 10/10/7 | 10/10/7 | 0 | 26.798 s | 11.625 s |
| beta_excel_multi_004 | existing six | 7/8/3 | 6/8/3 | 0 | 28.188 s | 17.257 s |
| beta_excel_multi_005 | new 21 | 6/7/3 | 6/8/4 | -1 | 37.927 s | 18.591 s |
| beta_cross_limit_001 | new 21 | 6/10/4 | 6/10/6 | -2 | 24.084 s | 7.671 s |
| beta_cross_limit_002 | existing six | 4/10/0 | 6/10/6 | -6 | 24.967 s | 9.694 s |

## Results by question family

| Family | Cases | DSH mean R/G/P | Original mean R/G/P | DSH better / tie / original better |
| --- | ---: | --- | --- | --- |
| User synthesis | 15 | 5.133/8.600/0.933 | 5.267/8.667/1.400 | 0 / 11 / 4 |
| PDF visual | 5 | 7.400/10.000/5.000 | 6.200/10.000/3.400 | 1 / 4 / 0 |
| Excel multi-table | 5 | 7.800/8.400/5.000 | 7.600/8.800/4.400 | 1 / 3 / 1 |
| Cross-source limitation | 2 | 5.000/10.000/2.000 | 6.000/10.000/6.000 | 0 / 0 / 2 |

DSH's value is concentrated rather than general:

- `beta_pdf_visual_002` improved from P 0 to 8 because DSH recovered the six
  requested fuse values; the original route refused for lack of context.
- `beta_excel_multi_001` improved from P 4 to 8 because DSH decomposed Table 1
  performance and Table 11 Territory spend into separate executable steps.
- On `beta_cross_limit_002`, DSH regressed from P 6 to 0 by calculating an
  unsupported percentage, while the original route safely abstained.
- The original route also handled the insufficient-context requirements in
  `beta_user_008`, `beta_user_014`, and `beta_cross_limit_001` more completely.

## Decision

The current broad `needs_planning()` gate is not selective enough to make DSH
the default for almost every complex question. In this run it doubled latency
for the newly tested 21 cases without improving any of their P scores.

Keep DSH as an optional planner behind the existing backend and unchanged
frontend, but narrow its activation to cases with a strong decomposition
benefit, especially explicit multi-table or multi-modal requests. Before wider
use, add:

1. a source-contradiction and missing-operand guard before calculation;
2. stricter modality, entity, table, period, and scope constraints;
3. a plan-value check that falls back to the original route when DSH produces
   no additional executable coverage;
4. repeated runs before treating the small aggregate score differences as
   statistically meaningful.

The answer model is not the primary bottleneck: both variants have mean G near
9, while mean P remains below 3. The next quality work should focus on evidence
coverage, verified binding, contradiction handling, and final requirement
coverage rather than frontend changes.
