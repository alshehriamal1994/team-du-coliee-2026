"""Paired bootstrap intervals for the paper's paired claims.

The abstract rests on "the paired improvement and not a ranking against the field", yet
expJ supplies marginal intervals only. Marginal intervals answer where each run's F1 lies;
they do not answer whether the improvement between two arms run on the same 100 cases is
resolved. The right instrument is a paired case-level bootstrap on the difference itself,
in which shared case difficulty cancels.

Three paired contrasts, each between runs on the identical 100 test cases:

  1. expO control vs treatment: the count clause alone, same day, same endpoint.
  2. expI v3-RAG with the count prior vs without it: the prior's worth.
  3. expO treatment vs expI v3-RAG without prior: count permission alone vs the full
     rewritten block, i.e. the broadened-wording share (between-day caveat applies).

Protocol. Resample the 100 case ids with replacement, B = 10,000, fixed seed. For each
resample recompute micro-averaged F1 for both arms over the resampled multiset and record
the difference. Report the mean difference, the percentile 95% interval and the fraction
of resamples in which the difference is positive.

Verification gate: the full-sample micro P, R and F1 of every arm must reproduce the
ledgered values (expO_numbers.json, expI_numbers.json) to four decimals, otherwise
nothing is written.

Writes expP_numbers.json.
"""
import json
import os
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")

import numpy as np

HERE = Path(__file__).parent
T2 = Path(ROOT) / "TASK2_code"
GOLD = Path(ROOT) / "task2_test_labels_2026(1).json"
OUT = HERE / "expP_numbers.json"
B = 10_000
SEED = 20260819

ARMS = {
    "control": T2 / "runs_singleclause/test2026_control.txt",
    "treatment": T2 / "runs_singleclause/test2026_treatment.txt",
    "v3_rag_prior": T2 / "runs_multiselect/test2026_v3_multiselect_rag.txt",
    "v3_rag_noprior": T2 / "runs_multiselect_noprior/test2026_v3_multiselect_rag.txt",
}
# (P, R, F1) as ledgered in expO_numbers.json and expI_numbers.json
EXPECT = {
    "control": (0.7528, 0.2279, 0.3499),
    "treatment": (0.6138, 0.3946, 0.4803),
    "v3_rag_prior": (None, None, 0.5487),
    "v3_rag_noprior": (None, None, 0.5183),
}
CONTRASTS = [
    ("count_clause_alone", "treatment", "control"),
    ("count_prior_worth", "v3_rag_prior", "v3_rag_noprior"),
    ("broadened_wording_share", "v3_rag_noprior", "treatment"),
]


def load_preds(path):
    preds = {}
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        cid, pid = parts[0], parts[1]
        if pid.strip("_").lower() != "none":
            preds.setdefault(cid, set()).add(pid)
    return preds


def load_gold():
    raw = json.loads(GOLD.read_text())
    gold = {}
    for k, v in raw.items():
        cid = k.replace(".txt", "")
        paras = v if isinstance(v, list) else [p.strip() for p in str(v).split(",")]
        gold[cid] = {p.replace(".txt", "").zfill(3) for p in paras}
    return gold


def micro(preds, gold, cases):
    tp = pred_n = gold_n = 0
    for c in cases:
        p = preds.get(c, set())
        g = gold[c]
        tp += len(p & g)
        pred_n += len(p)
        gold_n += len(g)
    if pred_n == 0 or gold_n == 0 or tp == 0:
        return 0.0, 0.0, 0.0
    prec, rec = tp / pred_n, tp / gold_n
    return prec, rec, 2 * prec * rec / (prec + rec)


def main():
    gold = load_gold()
    cases = sorted(gold)
    assert len(cases) == 100, f"expected 100 test cases, found {len(cases)}"
    arms = {name: load_preds(path) for name, path in ARMS.items()}

    gates = {}
    full = {}
    for name, preds in arms.items():
        p, r, f = micro(preds, gold, cases)
        full[name] = {"P": round(p, 4), "R": round(r, 4), "F1": round(f, 4)}
        ep, er, ef = EXPECT[name]
        ok = abs(f - ef) < 5e-5
        if ep is not None:
            ok = ok and abs(p - ep) < 5e-5 and abs(r - er) < 5e-5
        gates[f"{name}_reproduces_ledger"] = ok
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}\n{full}")

    rng = np.random.default_rng(SEED)
    idx = np.arange(100)
    contrasts = {}
    for label, a, b in CONTRASTS:
        diffs = np.empty(B)
        for t in range(B):
            sample = [cases[i] for i in rng.choice(idx, size=100, replace=True)]
            fa = micro(arms[a], gold, sample)[2]
            fb = micro(arms[b], gold, sample)[2]
            diffs[t] = fa - fb
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        contrasts[label] = {
            "arm_a": a, "arm_b": b,
            "full_sample_diff_pp": round(100 * (full[a]["F1"] - full[b]["F1"]), 2),
            "boot_mean_diff_pp": round(100 * float(diffs.mean()), 2),
            "boot_ci95_pp": [round(100 * float(lo), 2), round(100 * float(hi), 2)],
            "prob_positive": round(float((diffs > 0).mean()), 4),
            "excludes_zero": bool(lo > 0 or hi < 0),
        }

    res = {
        "experiment": "expP_paired_bootstrap",
        "purpose": "paired case-level bootstrap on the differences the paper claims",
        "protocol": {"B": B, "seed": SEED, "resampling_unit": "test case", "n_cases": 100},
        "gates": gates,
        "full_sample": full,
        "contrasts": contrasts,
        "note": ("The broadened_wording_share contrast crosses the expI/expO endpoint "
                 "dates and carries the between-day caveat stated in the paper."),
    }
    OUT.write_text(json.dumps(res, indent=2))
    for k, v in contrasts.items():
        print(f"  {k:26s} diff {v['full_sample_diff_pp']:+6.2f}pp  "
              f"95% CI [{v['boot_ci95_pp'][0]:+.2f}, {v['boot_ci95_pp'][1]:+.2f}]  "
              f"P(>0) {v['prob_positive']:.3f}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
