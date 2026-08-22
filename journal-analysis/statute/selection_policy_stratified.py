"""Stratified robustness check for the policy-level selection bootstrap (journal
article, Appendix).

The bootstrap of the selection procedure resamples the 258 validation questions
uniformly, which treats them as exchangeable. They come from three bar examinations with
an uneven label balance, so this script repeats the identical policy comparison under
schemes that hold that structure fixed: resampling within examination year, within gold
label, and within each year-by-label cell. Ensemble candidates, tie-breaking and
replicate count are identical across schemes.

Data loading and the policy machinery are imported from selection_policy_analysis.py, and
the script reproduces both that module's ledger gate and the published unstratified
figures exactly before computing any new quantity.

Writes selection_policy_stratified_numbers.json.
"""

import json
import sys
import os
from pathlib import Path

import numpy as np

import selection_policy_analysis as base

ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))
DATASETS = ROOT / "TASK4/experiments/datasets"
PUBLISHED = ROOT / "science/manuscript1_statute/selection_policy_numbers.json"
OUT = ROOT / "science/manuscript1_statute/selection_policy_stratified_numbers.json"

VAL_SPLITS = base.VAL_SPLITS
B_BOOT = base.B_BOOT
M_COMMITTEES = base.M_COMMITTEES
SEED = base.SEED           # reproduces the published draw exactly
SEED_STRAT = 20260804      # new stream for the stratified weights


def load_year_map():
    """Question id -> examination year, taken from the split files themselves.

    Not parsed from the id string: the R01 split mixes 'R01' and 'R1' id
    prefixes (64 and 46 respectively), so prefix parsing would silently split
    one examination year into two strata.
    """
    year = {}
    for split in VAL_SPLITS:
        with open(DATASETS / f"{split}_formal/test.jsonl") as f:
            for line in f:
                year[json.loads(line)["id"]] = split
    return year


def strata_indices(keys, val_ids):
    """Map stratum key -> array of positions in val_ids."""
    out = {}
    for i, q in enumerate(val_ids):
        out.setdefault(keys[q], []).append(i)
    return {k: np.array(v) for k, v in sorted(out.items())}


def stratified_weights(strata, n_val, b, rng):
    """(n_val, B) multinomial counts drawn independently within each stratum.

    Each stratum contributes exactly its own size to every replicate, so the
    stratum composition of every resampled validation set matches the observed
    one. With a single stratum covering everything this reduces exactly to the
    published unstratified scheme.
    """
    w = np.zeros((n_val, b), dtype=np.int64)
    for idx in strata.values():
        m = len(idx)
        w[idx, :] = rng.multinomial(m, np.full(m, 1 / m), size=b).T
    return w


def run_gate(pool, c_val, c_test):
    """selection_policy_analysis.py's own ledger gate, re-run here."""
    val_acc, test_acc = c_val.mean(axis=1), c_test.mean(axis=1)
    r = np.corrcoef(val_acc, test_acc)[0, 1]
    val_best = int(val_acc.argmax())
    du3_idx = np.array([[pool.index(e) for e in base.DU3_EXPERTS]])
    checks = {
        "pool_is_30": len(pool) == 30,
        "corr_0.43": round(float(r), 2) == 0.43,
        "val_best_is_llama33_standard":
            pool[val_best] == "llama-3.3-70b-instruct_standard_v1",
        "val_best_test_68_of_82": int(c_test[val_best].sum()) == 68,
        "val_best_rank28_tied_one_lower":
            int((test_acc > test_acc[val_best]).sum()) == 27
            and int((test_acc < test_acc[val_best]).sum()) == 1,
        "spread_13.4pp":
            round(float((test_acc.max() - test_acc.min()) * 100), 1) == 13.4,
        "du3_val_237_of_258":
            int(base.committee_correct(c_val, du3_idx)[0].sum()) == 237,
        "du3_test_77_of_82":
            int(base.committee_correct(c_test, du3_idx)[0].sum()) == 77,
    }
    for k, ok in checks.items():
        print(f"  gate {k}: {'PASS' if ok else 'FAIL'}")
    return all(checks.values())


def summary(dep, oracle_single):
    q = np.percentile(dep, [5, 50, 95]) * 100
    return {
        "mean_pc": round(float(dep.mean() * 100), 1),
        "sd_pp": round(float(dep.std() * 100), 1),
        "p5_pc": round(float(q[0]), 1),
        "median_pc": round(float(q[1]), 1),
        "p95_pc": round(float(q[2]), 1),
        "min_pc": round(float(dep.min() * 100), 1),
        "expected_regret_vs_oracle_pp":
            round(float((oracle_single - dep.mean()) * 100), 1),
        "prob_reach_du3_93.9": round(float((dep >= 77 / 82).mean()), 3),
        "prob_below_90": round(float((dep < 0.90).mean()), 3),
    }


def main():
    gold = base.load_gold()
    pool, c_val, c_test, val_ids, test_ids = base.build_pool(gold)
    n = len(pool)
    test_acc = c_test.mean(axis=1)
    oracle_single = test_acc.max()
    n_val = len(val_ids)

    print(f"pool: {n} experts, {n_val} val questions, "
          f"{len(test_ids)} test questions")
    if not run_gate(pool, c_val, c_test):
        print("LEDGER GATE FAILED - no new numbers computed.")
        sys.exit(1)

    # ---- reproduce the published draw exactly ------------------------------
    # Same seed and same draw ORDER as selection_policy_analysis.main():
    # weights first, then committees. Any deviation breaks reproduction.
    rng = np.random.default_rng(SEED)
    w_unstrat = rng.multinomial(n_val, np.full(n_val, 1 / n_val),
                                size=B_BOOT).T
    seen, members = set(), []
    while len(members) < M_COMMITTEES:
        c = tuple(sorted(rng.choice(n, size=9, replace=False)))
        if c not in seen:
            seen.add(c)
            members.append(c)
    members = np.array(members)
    comm_val = base.committee_correct(c_val, members)
    comm_test_acc = base.committee_correct(c_test, members).mean(axis=1)

    dep_single_u, _ = base.bootstrap_argmax_test_acc(c_val, test_acc, w_unstrat)
    dep_comm_u, _ = base.bootstrap_argmax_test_acc(comm_val, comm_test_acc,
                                                   w_unstrat)
    repro_single = summary(dep_single_u, oracle_single)
    repro_comm = summary(dep_comm_u, oracle_single)

    with open(PUBLISHED) as f:
        pub = json.load(f)
    repro_checks = {}
    for name, got, want in (("single", repro_single, pub["policy_single"]),
                            ("committee9", repro_comm, pub["policy_committee9"])):
        for key in ("mean_pc", "sd_pp", "p5_pc", "p95_pc",
                    "prob_below_90", "prob_reach_du3_93.9"):
            repro_checks[f"{name}.{key}"] = (got[key] == want[key], got[key],
                                             want[key])
    for k, (ok, got, want) in repro_checks.items():
        print(f"  repro {k}: {'PASS' if ok else f'FAIL got={got} want={want}'}")
    if not all(v[0] for v in repro_checks.values()):
        print("REPRODUCTION GATE FAILED - no new numbers computed.")
        sys.exit(1)
    print("  both gates PASS - computing stratified variants")

    # ---- strata ------------------------------------------------------------
    year = load_year_map()
    label = {q: gold[q] for q in val_ids}
    yl = {q: f"{year[q]}/{label[q]}" for q in val_ids}

    schemes = {
        "unstratified": {"all": np.arange(n_val)},
        "year": strata_indices(year, val_ids),
        "label": strata_indices(label, val_ids),
        "year_x_label": strata_indices(yl, val_ids),
    }
    for name, st in schemes.items():
        sizes = {k: int(len(v)) for k, v in st.items()}
        print(f"  scheme {name}: {len(st)} strata {sizes}")

    # ---- run every scheme on identical committees and policies -------------
    rng_s = np.random.default_rng(SEED_STRAT)
    results = {}
    for name, st in schemes.items():
        w = (w_unstrat if name == "unstratified"
             else stratified_weights(st, n_val, B_BOOT, rng_s))
        d_s, _ = base.bootstrap_argmax_test_acc(c_val, test_acc, w)
        d_c, _ = base.bootstrap_argmax_test_acc(comm_val, comm_test_acc, w)
        results[name] = {
            "strata": {k: int(len(v)) for k, v in st.items()},
            "n_strata": len(st),
            "policy_single": summary(d_s, oracle_single),
            "policy_committee9": summary(d_c, oracle_single),
            "committee_minus_single_mean_pp":
                round(float((d_c.mean() - d_s.mean()) * 100), 1),
            "prob_committee_beats_single":
                round(float((d_c > d_s).mean()), 3),
        }
        s, c = results[name]["policy_single"], results[name]["policy_committee9"]
        print(f"  {name:14s} single {s['mean_pc']:.1f} "
              f"(sd {s['sd_pp']:.1f}, P<90 {s['prob_below_90']:.2f})   "
              f"committee {c['mean_pc']:.1f} "
              f"(sd {c['sd_pp']:.1f}, P<90 {c['prob_below_90']:.2f})")

    out = {
        "protocol": {
            "purpose": "robustness of the Section 5.3 policy bootstrap to the "
                       "exchangeability assumption",
            "bootstrap_replicates": B_BOOT,
            "committee_candidates_sampled": M_COMMITTEES,
            "seed_published_draw": SEED,
            "seed_stratified_draws": SEED_STRAT,
            "committees_identical_across_schemes": True,
            "cluster_bootstrap_over_years": "not run; three examination years "
                                            "give a degenerate cluster "
                                            "bootstrap",
            "tie_break": "lowest index in alphabetical expert order",
        },
        "reproduction_gate": {
            "published_file": PUBLISHED.name,
            "all_checks_passed": True,
            "checked_keys": sorted(repro_checks),
        },
        "schemes": results,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
