# Journal-extension analyses (Tasks 3 and 4)

This folder contains the analysis code behind our journal extension of the
COLIEE 2026 proceedings paper, submitted to the Review of Socionetwork
Strategies special issue. The competition systems themselves live in the task
folders of this repository; everything here is post-hoc analysis of frozen
prediction files. No new model inference is required to reproduce any number.

## What is included

- The analysis scripts (Python 3, numpy and matplotlib only).
- The result files each script writes (`*_numbers.json`), as generated for
  the manuscript, so every figure in the paper can be traced to a value here.
- Our own model outputs under `data/`: the per-expert prediction files for
  the 30-expert pool on H30, R01, R02 and R07 (`data/TASK4/experiments/
  runs_ensemble/`, one line per question: id, predicted label, model tag),
  the nine experts' predictions on retrieved articles, and the submitted DU2
  retrieval run. These prediction files are outputs of our systems and contain
  no competition text. One exception for transparency: `example_boxes_facts.json`
  quotes, verbatim, the three test questions analysed as worked examples in the
  paper, exactly as the paper itself quotes them.

## What is not included

The COLIEE datasets (questions, gold labels, the Civil Code file) are
licensed by the organisers and are not redistributed. To rerun the scripts,
obtain the data from the organisers and place it under a directory pointed to
by the `COLIEE_ROOT` environment variable (default `data/`), with gold label
files at `TASK4/experiments/datasets/{H30,R01,R02}_formal/test.jsonl`,
question texts at `TASK4/experiments/datasets/test_R07/test.jsonl`, the
released R07 answers at `task3/QA.txt` (tab-separated id and label), and the
Civil Code at `civil.xml`.

## Verification gates

Each script begins by recomputing a set of previously established quantities
(pool size, the validation-best expert and its scores, the nine-expert vote
on all four splits) and aborts if any of them fails to reproduce. The
expected values are written into the scripts, so a silent change in the
inputs cannot produce silently different results.

## Script guide

| Script | Analysis in the paper |
|---|---|
| `selection_policy_analysis.py` | validation-to-test instability; the policy-level bootstrap; the risk-size curve |
| `gate1_audit.py` | gold provenance; frozen-submission rescoring; McNemar tests; the noise-only null |
| `diversity_robustness_analysis.py` | correlation intervals; diversity frontier with intervals; lift against structural and measured diversity |
| `equal_compute_analysis.py` | same-model prompt committees against multi-model committees at equal budget; selection stability across validation years |
| `margin_and_power_analysis.py` | vote margin as a confidence signal; benchmark-size projection; the winner's curse in the hindsight-best baseline |
| `design_rule_analysis.py` | the selection-reliability surface; the committee voting ceiling |
| `deltaq_ci_analysis.py` | oracle-versus-retrieved error-correlation interval |
| `task3_gap_analysis.py` | Task 3 error decomposition and leaderboard placement of the post-competition committee |
| `example_boxes.py` | the worked test-question examples |
| `review_response_analysis.py` | test-side bootstrap of the policy comparison and the headroom control for the lift-diversity correlation |
| `make_fig_selection.py`, `make_fig_policy.py`, `make_fig_design.py` | the paper's figures (`fig_style.py` holds the shared style) |

Scripts that depend on shared loaders import from
`selection_policy_analysis.py`, so run everything from this directory.
