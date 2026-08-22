"""Two further paired intervals for Task 2 (expP2, extending expP).

An outer-loop reviewer asked that every headline Task 2 comparison carry a paired
interval, not only the manipulation. Two comparisons lacked one:

  1. rewrite_total: the leakage-free rewritten instruction against the official
     submitted run, 0.338 to 0.518, on the same 100 cases (the between-endpoint
     caveat of Section 6.1 applies, since the official run predates the rerun).
  2. oracle_vs_fixed3: the true per-query count against the best fixed constant on
     the same fused ranking, 0.465 to 0.493.

Same design as expP: case-level bootstrap of the difference, B = 10,000, fixed seed.

Verification gates, before anything is written: the official DU2 file must reproduce
its official F1 = 0.3377; the leakage-free run must reproduce its ledgered 0.5183;
the fused ranking must reproduce fixed-3 = 0.465 and oracle = 0.4932.

Writes expP2_numbers.json.
"""
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")

import numpy as np

HERE = Path(__file__).parent
OFFICIAL = Path(ROOT) / "task2_DU2.txt"
NOPRIOR = Path(ROOT) / "test2026_v3_multiselect_rag.txt"
CACHE = Path(ROOT) / "test_cache_monot5v2.pkl"
LABELS = Path(ROOT) / "task2_test_labels_2026(1).json"
OUT = HERE / "expP2_numbers.json"
B = 10_000
SEED = 20260825


def load_preds(path):
    preds = defaultdict(set)
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].strip("_").lower() != "none":
            preds[parts[0]].add(parts[1].zfill(3))
    return preds


def main():
    raw = json.load(open(LABELS))
    gold = {c: {x.strip().replace(".txt", "").zfill(3) for x in v.split(",") if x.strip()}
            for c, v in raw.items()}
    cases = sorted(gold)
    total = sum(len(g) for g in gold.values())

    def micro(preds, sample=None):
        cs = sample if sample is not None else cases
        tp = sum(len(preds.get(c, set()) & gold[c]) for c in cs)
        n = sum(len(preds.get(c, set())) for c in cs)
        ng = sum(len(gold[c]) for c in cs)
        if n == 0 or ng == 0 or tp == 0:
            return 0.0
        P, R = tp / n, tp / ng
        return 2 * P * R / (P + R)

    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    ranking = {}
    for row in cache["rows"]:
        m5, q3 = np.array(row["m5"]), np.array(row["q3"])
        pids = [p.zfill(3) for p in row["cand_ids"]]
        r1, r2 = m5.max() - m5.min(), q3.max() - q3.min()
        n1 = np.ones_like(m5) if r1 < 1e-9 else (m5 - m5.min()) / r1
        n2 = np.ones_like(q3) if r2 < 1e-9 else (q3 - q3.min()) / r2
        order = np.argsort(-(0.8 * n1 + 0.2 * n2))
        ranking[row["cid"]] = [pids[i] for i in order]

    arms = {
        "official_DU2": load_preds(OFFICIAL),
        "noprior_rewrite": load_preds(NOPRIOR),
        "fixed3": {c: set(ranking[c][:3]) for c in cases},
        "oracle": {c: set(ranking[c][:len(gold[c])]) for c in cases},
    }
    full = {k: round(micro(v), 4) for k, v in arms.items()}
    gates = {
        "official_is_0.3377": abs(full["official_DU2"] - 0.3377) < 6e-4,
        "noprior_is_0.5183": abs(full["noprior_rewrite"] - 0.5183) < 6e-4,
        "fixed3_is_0.465": abs(full["fixed3"] - 0.4646) < 1e-3,
        "oracle_is_0.4932": abs(full["oracle"] - 0.4932) < 6e-4,
    }
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}\n{full}")

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(cases))
    contrasts = {}
    for label, a, b in [("rewrite_total", "noprior_rewrite", "official_DU2"),
                        ("oracle_vs_fixed3", "oracle", "fixed3")]:
        diffs = np.empty(B)
        for t in range(B):
            s = [cases[i] for i in rng.choice(idx, size=len(cases), replace=True)]
            diffs[t] = micro(arms[a], s) - micro(arms[b], s)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        contrasts[label] = {
            "full_sample_diff_pp": round(100 * (full[a] - full[b]), 2),
            "boot_ci95_pp": [round(100 * float(lo), 2), round(100 * float(hi), 2)],
            "prob_positive": round(float((diffs > 0).mean()), 4),
        }

    res = {"experiment": "expP2_paired_extra", "gates": gates, "full_sample": full,
           "protocol": {"B": B, "seed": SEED, "resampling_unit": "test case"},
           "contrasts": contrasts,
           "note": ("rewrite_total crosses the June-to-August endpoint gap; the "
                    "same-day manipulation of expO carries the caveat-free version")}
    OUT.write_text(json.dumps(res, indent=2))
    for k, v in contrasts.items():
        print(f"  {k:18s} {v['full_sample_diff_pp']:+6.2f}pp  "
              f"CI [{v['boot_ci95_pp'][0]:+.2f}, {v['boot_ci95_pp'][1]:+.2f}]  "
              f"P(>0) {v['prob_positive']}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
