# DSH planner evaluation (2026-08-23)

## Method

All 27 `beta_golden_questions` were generated blind: only the case ID,
category, and question entered the pipeline. The deterministic gate sent 25
questions with `needs_planning=true` to DSH and kept two simple questions on
the original route. DSH used local `qwen3:14b`; every final answer used
`deepseek/deepseek-v4-flash`.

DeepSeek then graded each saved result on three independent 0-10 dimensions:

- **R (retrieval):** task decomposition, source/modality selection, and
  retrieved evidence against the frozen requirements.
- **G (generation):** answer quality given only the evidence actually
  retrieved. Safe refusal or an explicit partial answer can receive 10/10 when
  evidence is insufficient.
- **P (pipeline):** end-to-end correctness against the frozen gold answer and
  expected behavior. Atomic requirement coverage caps this score.

This is one controlled run, not a statistical model benchmark.

## Full-set result

Mean scores were **R 6.037**, **G 8.926**, and **P 2.519**.

| Case | R | G | P |
| --- | ---: | ---: | ---: |
| beta_user_001 | 8 | 8 | 1 |
| beta_user_002 | 7 | 9 | 0 |
| beta_user_003 | 8 | 8 | 0 |
| beta_user_004 | 3 | 10 | 0 |
| beta_user_005 | 4 | 10 | 3 |
| beta_user_006 | 4 | 8 | 0 |
| beta_user_007 | 6 | 8 | 0 |
| beta_user_008 | 4 | 8 | 2 |
| beta_user_009 | 4 | 8 | 0 |
| beta_user_010 | 3 | 10 | 0 |
| beta_user_011 | 7 | 10 | 3 |
| beta_user_012 | 4 | 8 | 0 |
| beta_user_013 | 5 | 8 | 0 |
| beta_user_014 | 6 | 8 | 4 |
| beta_user_015 | 4 | 8 | 1 |
| beta_pdf_visual_001 | 10 | 10 | 8 |
| beta_pdf_visual_002 | 8 | 10 | 8 |
| beta_pdf_visual_003 | 7 | 10 | 6 |
| beta_pdf_visual_004 | 8 | 10 | 3 |
| beta_pdf_visual_005 | 4 | 10 | 0 |
| beta_excel_multi_001 | 10 | 10 | 8 |
| beta_excel_multi_002 | 6 | 7 | 4 |
| beta_excel_multi_003 | 10 | 10 | 7 |
| beta_excel_multi_004 | 7 | 8 | 3 |
| beta_excel_multi_005 | 6 | 7 | 3 |
| beta_cross_limit_001 | 6 | 10 | 4 |
| beta_cross_limit_002 | 4 | 10 | 0 |

The gap between G and P shows that final-answer behavior was usually sensible
for the supplied evidence, while retrieval coverage remained the dominant
bottleneck. The recurring failures were missing table/figure modalities,
incorrect Excel entity or row binding, missing regulator/guideline sources,
and an unsafe calculation when sources contradicted one another.

## Six-case original-route A/B

The same DeepSeek grader and rubric were applied to three high-scoring and
three low-scoring DSH cases rerun through the original planner.

| Case | DSH R/G/P | Original R/G/P | P delta |
| --- | --- | --- | ---: |
| beta_user_001 | 8/8/1 | 8/8/3 | -2 |
| beta_user_004 | 3/10/0 | 4/8/0 | 0 |
| beta_pdf_visual_002 | 8/10/8 | 2/10/0 | +8 |
| beta_excel_multi_001 | 10/10/8 | 8/10/4 | +4 |
| beta_excel_multi_004 | 7/8/3 | 4/8/0 | +3 |
| beta_cross_limit_002 | 4/10/0 | 6/10/6 | -6 |

DSH averaged **6.667/9.333/3.333** and the original route averaged
**5.333/9.000/2.167**. DSH improved three cases, tied one, and regressed two.
Mean latency was 32.020 seconds for DSH and 19.343 seconds for the original
route, making DSH about 65.5% slower on this subset.

The strongest DSH gains came from multi-modal and cross-table decomposition.
The largest regression was a cross-source contradiction: the original route
safely abstained, while the DSH route calculated an unsupported percentage.
The next quality work should therefore prioritize modality constraints, exact
entity/table binding, and a source-contradiction guard rather than replacing
the frontend or the answer schema.
