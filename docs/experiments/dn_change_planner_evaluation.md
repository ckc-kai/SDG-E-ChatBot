# `dn_change_planner` versus original pipeline (27-case A/B, 2026-08-23)

## Executive summary

The branch is materially faster, but it is not a clear end-to-end quality win.
Across all 27 questions, mean R/G/P changed from `5.926/9.037/2.667` to
`6.185/9.185/2.741`, while mean latency fell from 15.475 seconds to 10.864
seconds. However, the apparent P improvement comes from a
`needs_planning=false` question that did not use the planner. On the 25
questions that actually used the changed planner, mean P moved slightly down
from 2.440 to 2.360, with six wins, thirteen ties, and six losses.

The quality effect is concentrated by question family. Excel multi-table
questions improved from mean P 4.4 to 5.4, while user-synthesis questions fell
from 1.4 to 1.067. The answer model is not the main bottleneck: mean G remained
above 9, while omissions and incorrect evidence binding kept P low.

## Controlled configuration

The frozen 27-question `beta_golden_questions` set and the previously captured
original answers were reused. The branch was evaluated at commit `f6bd602`.

The planner was aligned to the tested original configuration rather than the
branch's example/default model:

- planner provider: Ollama;
- planner model: `qwen3:4b`, not `qwen3.5:9b`;
- planner temperature: `0`;
- planner output budget: `1200` tokens, intentionally increased from the
  original 500-token limit to test the branch's anti-truncation change;
- final answer provider/model: `deepseek/deepseek-v4-flash` with a 1500-token
  answer budget.

The branch provider fixes seed 42 as part of its reproducibility implementation;
the original baseline provider did not expose a seed. This is an implementation
difference worth retaining in the interpretation of this one-run comparison.

The deterministic `needs_planning()` gate sent 25 questions through the changed
planner. `beta_pdf_visual_001` and `beta_pdf_visual_004` stayed on the existing
non-planner route.

## Scoring

`deepseek/deepseek-v4-flash` graded every branch answer on the same independent
0-10 R/G/P rubric used for the original baseline:

- **R (retrieval):** decomposition, source/modality selection, entity-period
  binding, and retrieved evidence;
- **G (generation):** fidelity and safety given only the evidence supplied to
  generation. Missing retrieval is not penalized here; a justified partial
  answer or safe refusal can receive 10;
- **P (pipeline):** end-to-end correctness against the frozen expected answer
  and behavior.

Both variants used the same frozen atomic-requirement checklist for each case.
P was capped by the proportion of requirements backed by a short verbatim
excerpt from the final answer. The grader used a 4000-token output budget and
temperature 0.

## Full 27-case result

| Metric | `dn_change_planner` | Original | Difference |
| --- | ---: | ---: | ---: |
| Mean R | 6.185 | 5.926 | +0.259 |
| Mean G | 9.185 | 9.037 | +0.148 |
| Mean P | 2.741 | 2.667 | +0.074 |
| Mean latency | 10.864 s | 15.475 s | -4.611 s (-29.8%) |
| Median latency | 10.256 s | 16.010 s | -5.754 s (-35.9%) |
| Total pipeline time | 293.338 s | 417.827 s | -124.489 s |
| P wins / ties / losses | 7 / 14 / 6 | — | — |

The branch produced two passes, ten partial results, and fifteen failures. The
original produced one pass, fourteen partial results, and twelve failures. The
slightly higher mean P therefore reflects a few larger wins, not uniformly
better reliability.

## Planner-only 25-case result

The two `needs_planning=false` cases are excluded here so that normal answer
variation on the unchanged route is not credited to the planner.

| Metric | `dn_change_planner` | Original | Difference |
| --- | ---: | ---: | ---: |
| Mean R | 5.960 | 5.680 | +0.280 |
| Mean G | 9.120 | 8.960 | +0.160 |
| Mean P | 2.360 | 2.440 | -0.080 |
| Mean latency | 11.343 s | 16.306 s | -4.963 s (-30.4%) |
| Median latency | 10.464 s | 16.129 s | -5.665 s (-35.1%) |
| Total pipeline time | 283.582 s | 407.638 s | -124.056 s |
| P wins / ties / losses | 6 / 13 / 6 | — | — |

The changed planner improves retrieval score and speed, but the extra retrieval
quality does not yet survive the final requirement-coverage step. On the cases
that actually exercised it, end-to-end answer quality is effectively neutral
to slightly lower in this single run.

## Results by question family

| Family | Cases | Branch mean R/G/P | Original mean R/G/P | Branch better / tie / original better |
| --- | ---: | --- | --- | --- |
| User synthesis | 15 | 5.400 / 8.667 / 1.067 | 5.267 / 8.667 / 1.400 | 3 / 7 / 5 |
| PDF visual | 5 | 6.600 / 10.000 / 4.200 | 6.200 / 10.000 / 3.400 | 1 / 4 / 0 |
| Excel multi-table | 5 | 7.800 / 9.600 / 5.400 | 7.600 / 8.800 / 4.400 | 2 / 3 / 0 |
| Cross-source limitation | 2 | 7.000 / 10.000 / 5.000 | 6.000 / 10.000 / 6.000 | 1 / 0 / 1 |

The PDF-family gain includes `beta_pdf_visual_004`, which was routed through
the unchanged `needs_planning=false` path. It is run-to-run answer variation,
not evidence that the planner improved visual retrieval. Excel multi-table is
the clearest family-level benefit; broad user synthesis remains the main
quality weakness.

## Per-answer quality

`P delta` is branch minus original. `Pass`, `partial`, and `fail` are determined
from calibrated P (`>=8`, `>=3`, and `<3`).

| Case | Route | Branch R/G/P | Original R/G/P | P delta | Branch quality | Branch time | Original time |
| --- | --- | --- | --- | ---: | --- | ---: | ---: |
| `beta_user_001` | planner | 8/9/2 | 8/8/3 | -1 | fail, regressed | 15.563 s | 31.381 s |
| `beta_user_002` | planner | 2/10/0 | 7/8/1 | -1 | fail, regressed | 7.539 s | 32.872 s |
| `beta_user_003` | planner | 6/8/0 | 7/10/0 | 0 | fail, tied | 8.187 s | 35.899 s |
| `beta_user_004` | planner | 4/9/0 | 4/8/0 | 0 | fail, tied | 18.913 s | 17.656 s |
| `beta_user_005` | planner | 4/7/0 | 6/10/3 | -3 | fail, regressed | 10.633 s | 9.414 s |
| `beta_user_006` | planner | 6/7/0 | 4/8/0 | 0 | fail, tied | 10.464 s | 11.158 s |
| `beta_user_007` | planner | 8/10/1 | 6/8/0 | +1 | fail, improved | 8.086 s | 15.654 s |
| `beta_user_008` | planner | 6/10/4 | 4/10/4 | 0 | partial, tied | 6.835 s | 17.129 s |
| `beta_user_009` | planner | 4/8/1 | 7/8/0 | +1 | fail, improved | 12.386 s | 18.318 s |
| `beta_user_010` | planner | 5/8/0 | 3/10/0 | 0 | fail, tied | 17.411 s | 16.129 s |
| `beta_user_011` | planner | 4/8/1 | 4/10/3 | -2 | fail, regressed | 10.523 s | 9.069 s |
| `beta_user_012` | planner | 6/8/0 | 4/8/0 | 0 | fail, tied | 10.256 s | 11.717 s |
| `beta_user_013` | planner | 8/10/1 | 7/8/0 | +1 | fail, improved | 7.740 s | 15.375 s |
| `beta_user_014` | planner | 6/10/6 | 4/8/6 | 0 | partial, tied | 6.335 s | 16.992 s |
| `beta_user_015` | planner | 4/8/0 | 4/8/1 | -1 | fail, regressed | 16.377 s | 16.958 s |
| `beta_pdf_visual_001` | original route | 10/10/8 | 10/10/8 | 0 | pass, tied | 6.818 s | 7.218 s |
| `beta_pdf_visual_002` | planner | 2/10/0 | 2/10/0 | 0 | fail, tied | 7.090 s | 23.656 s |
| `beta_pdf_visual_003` | planner | 7/10/6 | 7/10/6 | 0 | partial, tied | 5.949 s | 6.119 s |
| `beta_pdf_visual_004` | original route | 8/10/7 | 8/10/3 | +4 | partial, improved | 2.938 s | 2.971 s |
| `beta_pdf_visual_005` | planner | 6/10/0 | 4/10/0 | 0 | fail, tied | 4.450 s | 4.882 s |
| `beta_excel_multi_001` | planner fallback | 10/10/8 | 8/10/4 | +4 | pass, improved | 21.459 s | 16.412 s |
| `beta_excel_multi_002` | planner | 7/8/4 | 8/8/4 | 0 | partial, tied | 14.082 s | 16.010 s |
| `beta_excel_multi_003` | planner | 10/10/7 | 10/10/7 | 0 | partial, tied | 16.170 s | 11.625 s |
| `beta_excel_multi_004` | planner | 8/10/4 | 6/8/3 | +1 | partial, improved | 9.874 s | 17.257 s |
| `beta_excel_multi_005` | planner | 4/10/4 | 6/8/4 | 0 | partial, tied | 14.874 s | 18.591 s |
| `beta_cross_limit_001` | planner | 7/10/7 | 6/10/6 | +1 | partial, improved | 9.525 s | 7.671 s |
| `beta_cross_limit_002` | planner | 7/10/3 | 6/10/6 | -3 | partial, regressed | 12.861 s | 9.694 s |

## Important qualitative cases

- **`beta_excel_multi_001` (+4 P):** the branch answer included target, actual,
  status, percent complete, Territory CAPEX/OPEX, and combined spend in both
  requested units. The original omitted all spend fields. This case used the
  branch fallback plan rather than a successfully parsed model plan, so the win
  cannot be attributed to the 1200-token budget alone.
- **`beta_user_005` (-3 P):** the branch answer produced unsupported SDG&E and
  SCE completion percentages and only marked PG&E as unavailable. The original
  safely declined the comparison. This is the most serious regression because
  it is a fabrication rather than a simple omission.
- **`beta_user_002` (R -5, G 10, P 0):** the branch safely refused because the
  supplied evidence did not support a quantified cross-cycle analysis. That
  refusal deserves a high G score, but retrieval missed evidence that enabled
  the original to give a scoped partial answer, so the pipeline score remained
  zero.
- **`beta_cross_limit_002` (-3 P):** the branch correctly abstained from the
  percentage, but it failed to recover the supported substation-inspection
  operand that the original answer included. G stayed 10 because the answer did
  not invent missing evidence; P fell because the end-to-end answer was less
  complete.
- **`beta_pdf_visual_004` (+4 P):** the branch run reported all four requested
  years, while the original missed 2024 and incorrectly treated 2026 as absent.
  This question did not use the planner, so it is excluded from claims about the
  planner change.

## Planner diagnostics and 1200-token observation

Of the 25 planned cases, 24 produced a model plan and one
(`beta_excel_multi_001`) used the branch fallback. No case recorded a planner
retry or a dropped atomic task. The 1200-token setting therefore ran through the
entire benchmark without a recorded plan-task truncation. This is useful
operational evidence for keeping the larger budget, but it is not an isolated
500-versus-1200 A/B and does not by itself prove that the larger budget caused
the quality changes.

On a fresh resume process, parallel retrieval attempted to initialize the
reranker singleton from more than one thread and failed before saving the next
case. The evaluation harness then pre-warmed the same reranker once and resumed
successfully; no production code was modified for this workaround. The branch
still needs a thread-safe cold-start initialization guard before deployment.

## Decision

Keep the 1200-token planner budget and the branch's latency improvements, but do
not claim a broad answer-quality improvement yet. The most defensible result is:

1. planner-only P is essentially neutral (`-0.080`) while latency improves by
   about 30%;
2. Excel multi-table quality improves, but generic synthesis quality regresses;
3. high G and low P show that final requirement coverage and evidence binding,
   not the DeepSeek answer model, remain the dominant bottlenecks;
4. the existing frontend can remain unchanged because all changes stay behind
   the current backend response contract.

Before enabling this planner broadly, add a thread-safe reranker cold-start,
guards against unsupported cross-utility percentages, and a final
requirement-coverage pass that either fills supported omissions or explicitly
marks them missing. A repeated same-branch 500-versus-1200 test would isolate
the anti-truncation benefit from the other batching and tracing changes.

## Validity limits

This is one paired run, not a statistical benchmark. The original answers were
captured in two earlier batches, and model responses can vary between runs. The
branch seed behavior differs from the original implementation. Latency was
measured on the same host but includes different warm-cache states; it should be
read as directional. The frozen questions, final answer model, DeepSeek grader,
atomic checklists, and score calibration were otherwise held constant.
