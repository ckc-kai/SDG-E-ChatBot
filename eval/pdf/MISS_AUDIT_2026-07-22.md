# Text Retrieval Hit@1 Miss Audit — 2026-07-22

## Scope and adjudication rule

This audit compares the actual top-ranked chunk with the labeled gold evidence for all 47 hit@1 misses from `2026-07-21-off-metric10-candidates30.json`.

A retrieved chunk is accepted as an alternative only when it independently contains enough information to answer the question. A neighboring or topically related chunk remains a miss when it omits a requested number, method, condition, or list item.

The audit is text-retrieval-only. It does not change chunking, embeddings, candidate generation, reranking, or structured-content extraction.

## Adjudication summary

- 21 misses had an independently sufficient top chunk and now have two alternatives in one required evidence group.
- 1 miss (`0128`) had an incorrect continuation-chunk label; the complete preceding chunk replaces it.
- 1 miss (`0021`) had an ambiguous question because several stakeholder recommendations modified SDGE-23-06; the question now identifies MGRA's falling-conductor recommendation.
- 24 misses are genuine retrieval or reranking errors and remain misses.

## Independently sufficient alternatives

| ID | Why the former top result is acceptable |
|---|---|
| 0009 | Appendix F states the same maturity range, 0 through 4. |
| 0010 | The same-section chart text and its footnote establish that SDG&E reported zero catastrophic wildfires. |
| 0012 | The required-progress summary directly states that SDG&E prioritizes undergrounding without adequate justification. |
| 0017 | The executive overview lists fire-detection automation, sensor and smoke-plume work, and machine-learning wind models. |
| 0019 | The executive summary explicitly describes the decision as Energy Safety's assessment and approval. |
| 0024 | Appendix F directly explains that maturity is scored from 0 to 4. |
| 0029 | The required-area summary contains all three requested maintenance-planning factors. |
| 0032 | The executive overview independently lists the requested situational-awareness and fire-detection improvements. |
| 0037 | The assessment section directly gives 37 capabilities and seven categories. |
| 0045 | The continuation explains how new information and technology generate proposed mitigations that are reviewed for feasibility, costs, and benefits. |
| 0049 | The model documentation directly states WiNGS-Planning's purpose. |
| 0056 | The continuation directly identifies cultural, water, and biological resource constraints. |
| 0058 | The AFN plan describes portable-generator/power-station rebates and fixed solar-plus-battery backup for PSPS-prone communities. |
| 0059 | The program overview directly states the wildfire-risk and PSPS-reduction purposes of undergrounding. |
| 0099 | The communication section identifies first responders, jurisdictions, tribes, water and telecommunications providers, and emergency agencies as public safety partners. |
| 0103 | The prioritization section directly describes binning the riskiest overhead circuit segments to show risk distribution. |
| 0133 | The neighboring table continuation directly reports the approximately 60% reduction. |
| 0139 | The AFN support section directly names the food-bank, Feeding America, Meals on Wheels, and other food partnerships. |
| 0143 | The AFN plan independently lists PSPS exercises and responder training. |
| 0147 | The decision executive summary explicitly gives the March 27, 2023 submission date. |
| 0150 | The standalone errata table directly lists the corrected program tracking IDs and affected programs. |

## Corrected label and question

| ID | Correction |
|---|---|
| 0128 | Replaced chunk 1 with chunk 0. Chunk 0 contains the complete `2.4.1` sample row and `20.8%`; chunk 1 begins after the identifying cells. |
| 0021 | Narrowed the question to MGRA's recommendation to add falling-conductor protection. The former wording admitted several different SDGE-23-06 modifications as answers. |

## Genuine misses retained

| ID | Why the top result remains insufficient |
|---|---|
| 0007 | Gives a high-level investment summary but not the complete requested hardening component set. |
| 0016 | Gives normalized PSPS metrics, not the requested absolute customer targets. |
| 0027 | Describes WiNGS-Planning but omits the time-value-of-risk bias identified by Energy Safety. |
| 0052 | Explains the 4 kV/12 kV connectivity change and 206 segments but omits the requested average 0.16 reduction. |
| 0057 | Discusses historical wildfire lessons, not the emergency-preparedness objectives. |
| 0062 | Discusses climate-risk modeling, not how new traditional hardening work is scoped. |
| 0064 | Describes a DCRI target change but not why the program lacks a quantified risk-reduction estimate. |
| 0065 | Describes general AFN identification, not Standby Power Program selection by meter, circuit, and PSPS exposure. |
| 0066 | Describes the distribution inspection program, not the requested transmission program. |
| 0071 | Discusses other continued-improvement technologies, not the expulsion-fuse completion date and efficacy study. |
| 0076 | Returns glossary material, not the inspection and cross-business process for identifying new subject poles. |
| 0088 | Describes customer support representatives, not centralized EOC/ICS emergency coordination. |
| 0093 | Gives general risk-model lessons, not the post-2020 weather normalization study. |
| 0095 | Covers event duration and active fires but omits the requested percentile-wind calculation and protocol relationship. |
| 0107 | Gives a high-level investment slide, not the method used to track PSPS risk reduction. |
| 0116 | Gives future joint-study requirements, not how tested covered-conductor effectiveness values were updated. |
| 0122 | The continuation discusses corrosion but omits the requested 10–15 cm distance. |
| 0123 | Says only “electrochemical testing”; it omits the requested cyclic-polarization method. |
| 0129 | Gives slip-load results but not the July 26, 2022 completion date. |
| 0137 | Contains an incomplete per-utility table and does not provide the requested statewide total of approximately 3.8 million. |
| 0138 | Discusses outreach resources, not the Joint IOU planning assumptions. |
| 0142 | Describes EOC preparation and training, not PSPS Working Group membership and 2022 topics. |
| 0145 | Describes an AFN council meeting but omits the outreach totals of about 360 events, 90 presentations, and 5,100 social posts. |
| 0148 | Says the SCADA section number was corrected but omits the corrected number, 8.1.4.3. |

## Corrected baseline

Run configuration: query rewrite off, dense candidate count 30, reranking and metrics through rank 10.

| Metric | Before audit | After audit |
|---|---:|---:|
| hit@1 | 0.6867 | 0.8400 |
| recall@5 | 0.8567 | 0.9100 |
| recall@10 | 0.8833 | 0.9300 |
| MRR@10 | 0.7589 | 0.8740 |
| nDCG@10 | 0.7872 | 0.8854 |

After correction, 24 hit@1 misses remain. Nine have no acceptable gold evidence in the 30 dense candidates. Of the other 15, the acceptable gold group is reranked to position 2 in seven cases, position 3 in two, position 4 in two, position 6 in two, position 10 in one, and position 14 in one.
