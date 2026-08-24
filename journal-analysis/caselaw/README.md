# Case law journal analysis (Tasks 1 and 2)

Analysis scripts and result ledgers behind the journal article "Learning to
Rank and Learning to Stop: What Each Is Worth in the COLIEE 2026 Legal Case
Retrieval and Entailment Tasks". Each `exp*_numbers.json` is the ledger a reported number traces
to, written by the `exp*.py` script beside it, except where noted below. Scripts that recompute from raw
predictions carry verification gates and abort rather than write if a published
anchor fails to reproduce.

`expH_runs.csv` holds the answer-set sizes of all 76 scored official runs,
derived from published precision and recall as described in the article.

The COLIEE data are available from the competition organisers under their
data-use agreement and are not redistributed here. Scripts that read raw
predictions or caches resolve them under the directory named by the
`COLIEE_ROOT` environment variable, which defaults to `data/`. The five
API-calling scripts read `OPENROUTER_API_KEY` from the environment and contain
no credentials. The worked-example cache shipped here holds identifiers and
retrieval scores only, the case text being licensed and read from your own copy
under `COLIEE_ROOT`.

## Script guide

- `expA_t1_cardinality.py` — Task 1 protocol ceilings and per-cardinality decomposition at fixed k=5.
- `expA2_t1_ladder.py` — full DU9 rankings: fixed-k sweep, oracle count, cardinality ladder.
- `expB_*` — evidence-depth measurement over the 1,750 gold pairs.
- `expC_cardinality_sweep.py` — Task 2 fixed-count and threshold sweeps.
- `expD*, expE*, expF*, expG*` — confidence elicitation, cross-year transfer, bare top-1 baselines, learned count predictors (cross-fit).
- `expH_leaderboard_policy.py`, `expH2_leaderboard_null.py` — the 2026 leaderboard audit and its permutation nulls.
- `expI` (ledger) — the leakage-free multi-selection re-run (prior rule deleted).
- `expJ_bootstrap_leakagefree.py` — marginal case-level bootstrap for the leakage-free runs.
- `expK` (ledger) — the Arampatzis score-distribution method on Task 1.
- `expL_decomposition.py` — the four-segment score decomposition of Figure 4.
- `expM`, `expN` — the count-value-versus-ranker-quality simulation and its real-ranker series.
- `expP_paired_bootstrap.py`, `expP2_paired_extra.py` — paired case-level bootstraps for every headline Task 2 difference.
- `expQ_replication_2025.py` with `run_replication_2025.py`, `replication2025_untouched_ids.json`, `bm25_top20_2025.json`, `fewshot_cache_2025.json` — the frozen count-clause manipulation replicated on the 71 untouched COLIEE 2025 cases through an untrained pipeline.
- `expR_t1_bootstrap.py` — query-level bootstrap intervals for the Task 1 decomposition and the peak-at-five claim.
- `expT_t2_count_predictor.py` — the Task 2 learned count predictor (trained before the shift; worse than the best constant).
- `expU_ceiling_trend.py` — the 2023-2026 answer-count distribution and fixed-count ceiling trend.
- `expV_leaderboard_2024.py` — the leaderboard audit replicated on the 2024 edition (association absent there).
- `expW_lecard.py` with `lecard_data/` — the count-value relation on LeCaRD's published baseline rankings (data MIT-licensed, from github.com/myx666/LeCaRD; licence included).
- `expX_team_level_null.py` -> `expX_numbers.json`. Applies the two permutation
  nulls of `expH2` to the best-run-per-team subsets, so the non-independence check
  is read on the same scale as the full-field association.
- `make_fig_*.py`, `make_figures2.py`, `figstyle.py` — figure generators.
- `make_appendix_audit.py` — Appendix C from `expH_runs.csv`.
- `expF_verify.py` checks the internal consistency of `expF_numbers.json` and the
  difference column the article prints from it.
- `expI_numbers.json` and `expO_numbers.json` are ledgers
  only. They record runs made before this folder was assembled, and the figures they
  hold are reproduced by the scripts that cite them rather than by a generator of
  their own.

- `expY_citation_treatment.py` -> `expY_numbers.json`. Counts query cases where a
  citation-suppression marker stands near distinguishing language, with a
  random-offset placebo for the ambient rate.
- `expZ_reference_ranker_sweep.py` -> `expZ_numbers.json`. The six post-competition
  Task 1 configurations and their test scores, recording how the reference ranker
  was selected.
- `expH3_exact_envelope.py` -> `expH3_numbers.json`. Null B recomputed with the
  exact perfect-ranker envelope, gated on reproducing expH2 first.
- `expS_submitted_oracle.py` -> `expS_numbers.json`. Rebuilds the submitted DU3
- `expAB_conformal.py` -> `expAB_numbers.json` — split conformal prediction on the fused reranker score, 825 pre-test calibration cases. Coverage holds on the held-out development split and collapses on the shifted test collection, as registered in the script docstring before the run.
- `expAC_containment.py` -> `expAC_numbers.json` — does the count permission change which paragraphs are chosen, or only how many. Wherever both arms selected, the treatment set contains the control's choice.
- `expAD_config_robustness.py` -> `expAD_numbers.json` — the count's worth re-evaluated on all six post-competition Task 1 configurations through the deployed postprocessing. The band runs 5.65 to 6.87 points.
  booster, gated on reproducing its official run exactly, and prices the answer
  count on that ranking rather than on the post-competition reference.
- `run_prompt_robustness.py` -> `expAA_numbers.json`. The count permission across
  three wordings and three runs each, against a shared control extracted verbatim
  from the single-clause manipulation.
