"""Selection risk as a policy-level quantity (Manuscript 1, Section 5).

Quantifies the deployment risk of model selection on the COLIEE 2026 Task 4
expert pool by bootstrapping the selection procedure itself. Two policies are
compared over resampled validation sets: deploying the validation-best single
expert, and deploying a nine-expert majority-vote committee chosen by the same
validation-argmax procedure. Also computes the risk-size curve for random
k-expert committees.

Before any new quantity is computed, the script must reproduce the verified
ledger in deep_analysis_numbers.json exactly (pool size, val-test correlation,
the validation-best expert and its test rank, DU3 vote accuracies). It aborts
if any check fails.

Inputs (read-only):
  TASK4/experiments/runs_ensemble/{split}.{expert}   per-expert predictions
  TASK4/experiments/datasets/{split}_formal/test.jsonl  validation gold
  <COLIEE_ROOT>/task3/QA.txt                          released R07 gold

Outputs (this folder):
  selection_policy_numbers.json
  figures/fig_selection_policy.png  (built separately by make_fig_policy.py)
"""

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))
RUNS = ROOT / "TASK4/experiments/runs_ensemble"
DATASETS = ROOT / "TASK4/experiments/datasets"
R07_GOLD = ROOT / "task3" / "QA.txt"  # released gold answers, from the organisers
OUT = Path(__file__).parent / "selection_policy_numbers.json"

VAL_SPLITS = ["H30", "R01", "R02"]
ALL_SPLITS = VAL_SPLITS + ["R07"]

# fine-tuned encoders, meta-ensembles and derived runs are not single experts
EXCLUDE_PREFIXES = (
    "bert", "ce-base", "deberta", "mbert", "deliberation", "neural",
    "stacking",
)

DU3_EXPERTS = [
    "deepseek-r1-zs",
    "llama-4-maverick_standard_v1",
    "llama-4-scout_standard_v1",
    "llama-3.3-70b-instruct_concise_v1",
    "llama-3.3-70b-instruct_cot_strict_v1",
    "qwen-2.5-72b-instruct_cot_strict_v1",
    "qwen3-235b-a22b_standard_v1",
    "qwen3-235b-a22b_irac_v1",
    "qwen3-235b-a22b_sc3_v1",
]

B_BOOT = 10_000
M_COMMITTEES = 5_000
M_SIZE = 2_000
K_GRID = [1, 3, 5, 7, 9, 11, 13, 15]
SEED = 20260704


def load_gold():
    gold = {}
    for split in VAL_SPLITS:
        with open(DATASETS / f"{split}_formal/test.jsonl") as f:
            for line in f:
                d = json.loads(line)
                gold[d["id"]] = d["label"]
    with open(R07_GOLD) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                gold[parts[0]] = parts[1]
    return gold


def load_predictions(expert, split):
    preds = {}
    path = RUNS / f"{split}.{expert}"
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[1] in ("Y", "N"):
                preds[parts[0]] = parts[1]
    return preds


def build_pool(gold):
    """Experts with complete Y/N predictions on all four splits."""
    names = sorted({p.name.split(".", 1)[1] for p in RUNS.iterdir()})
    names = [n for n in names if not n.startswith(EXCLUDE_PREFIXES)]

    val_ids = sorted(q for q in gold if not q.startswith("R07"))
    test_ids = sorted(q for q in gold if q.startswith("R07"))

    pool, c_val, c_test = [], [], []
    for name in names:
        preds = {}
        ok = True
        for split in ALL_SPLITS:
            p = load_predictions(name, split)
            if p is None:
                ok = False
                break
            preds.update(p)
        if not ok:
            continue
        if any(q not in preds for q in val_ids + test_ids):
            continue
        pool.append(name)
        c_val.append([preds[q] == gold[q] for q in val_ids])
        c_test.append([preds[q] == gold[q] for q in test_ids])

    return pool, np.array(c_val), np.array(c_test), val_ids, test_ids


def committee_correct(c, members_idx):
    """Majority-vote correctness matrix for committees over questions.

    For binary labels the vote equals gold exactly when more than half the
    members are individually correct on that question.
    """
    k = members_idx.shape[1]
    counts = c[members_idx].sum(axis=1)  # (M, n_questions)
    return counts > (k / 2)


def bootstrap_argmax_test_acc(c_val, test_acc, weights):
    """Deployed test accuracy when the validation-argmax row is selected.

    weights: (n_val, B) multinomial question counts per bootstrap replicate.
    Ties broken by lowest row index (fixed, documented).
    """
    deployed = np.empty(weights.shape[1])
    sel = np.empty(weights.shape[1], dtype=int)
    chunk = 1_000
    cf = c_val.astype(np.float32)
    for start in range(0, weights.shape[1], chunk):
        w = weights[:, start:start + chunk].astype(np.float32)
        scores = cf @ w  # (rows, chunk)
        idx = scores.argmax(axis=0)
        sel[start:start + chunk] = idx
        deployed[start:start + chunk] = test_acc[idx]
    return deployed, sel


def main():
    rng = np.random.default_rng(SEED)
    gold = load_gold()
    pool, c_val, c_test, val_ids, test_ids = build_pool(gold)

    n = len(pool)
    val_acc = c_val.mean(axis=1)
    test_acc = c_test.mean(axis=1)

    print(f"pool: {n} experts, {len(val_ids)} val questions, "
          f"{len(test_ids)} test questions")

    # ---- verification gate against deep_analysis_numbers.json -------------
    r = np.corrcoef(val_acc, test_acc)[0, 1]
    val_best = int(val_acc.argmax())
    # 27 experts strictly higher, 1 tied, 1 strictly lower: rank 28 of 30.
    # The earlier ledger's "29 of 30" was an argsort artefact among the tie.
    n_strictly_higher = int((test_acc > test_acc[val_best]).sum())
    n_strictly_lower = int((test_acc < test_acc[val_best]).sum())
    spread = (test_acc.max() - test_acc.min()) * 100
    du3_idx = np.array([[pool.index(e) for e in DU3_EXPERTS]])
    du3_val = committee_correct(c_val, du3_idx)[0].sum()
    du3_test = committee_correct(c_test, du3_idx)[0].sum()

    checks = {
        "pool_is_30": n == 30,
        "corr_0.43": round(float(r), 2) == 0.43,
        "val_best_is_llama33_standard":
            pool[val_best] == "llama-3.3-70b-instruct_standard_v1",
        "val_best_test_68_of_82": int(c_test[val_best].sum()) == 68,
        "val_best_rank28_tied_one_lower":
            n_strictly_higher == 27 and n_strictly_lower == 1,
        "spread_13.4pp": round(float(spread), 1) == 13.4,
        "du3_val_237_of_258": int(du3_val) == 237,
        "du3_test_77_of_82": int(du3_test) == 77,
    }
    for k, ok in checks.items():
        print(f"  gate {k}: {'PASS' if ok else 'FAIL'}")
    if not all(checks.values()):
        print("VERIFICATION GATE FAILED - no new numbers computed.")
        print(f"  n={n} r={r:.4f} val_best={pool[val_best]} "
              f"test={c_test[val_best].sum()}/82 higher={n_strictly_higher} "
              f"lower={n_strictly_lower} "
              f"spread={spread:.1f} du3={du3_val}/258,{du3_test}/82")
        sys.exit(1)

    n_val, n_test = len(val_ids), len(test_ids)
    oracle_single = test_acc.max()

    # ---- policy 1: deploy the validation-best single expert ---------------
    weights = rng.multinomial(n_val, np.full(n_val, 1 / n_val),
                              size=B_BOOT).T  # (n_val, B)
    dep_single, sel_single = bootstrap_argmax_test_acc(c_val, test_acc,
                                                       weights)

    # ---- policy 2: deploy the validation-best nine-expert committee -------
    seen = set()
    members = []
    while len(members) < M_COMMITTEES:
        c = tuple(sorted(rng.choice(n, size=9, replace=False)))
        if c not in seen:
            seen.add(c)
            members.append(c)
    members = np.array(members)
    comm_val = committee_correct(c_val, members)
    comm_test_acc = committee_correct(c_test, members).mean(axis=1)
    dep_comm, sel_comm = bootstrap_argmax_test_acc(comm_val, comm_test_acc,
                                                   weights)
    r_comm = np.corrcoef(comm_val.mean(axis=1), comm_test_acc)[0, 1]

    def summary(dep):
        q = np.percentile(dep, [5, 25, 50, 75, 95]) * 100
        return {
            "mean_pc": round(float(dep.mean() * 100), 1),
            "sd_pp": round(float(dep.std() * 100), 1),
            "p5_pc": round(float(q[0]), 1),
            "median_pc": round(float(q[2]), 1),
            "p95_pc": round(float(q[4]), 1),
            "min_pc": round(float(dep.min() * 100), 1),
            "expected_regret_vs_oracle_pp":
                round(float((oracle_single - dep.mean()) * 100), 1),
            "prob_reach_du3_93.9": round(float((dep >= 77 / 82).mean()), 3),
            "prob_below_90": round(float((dep < 0.90).mean()), 3),
        }

    # ---- risk-size curve ---------------------------------------------------
    size_curve = {}
    for k in K_GRID:
        if k == 1:
            accs = test_acc
        else:
            seen_k = set()
            mk = []
            while len(mk) < M_SIZE:
                c = tuple(sorted(rng.choice(n, size=k, replace=False)))
                if c not in seen_k:
                    seen_k.add(c)
                    mk.append(c)
            accs = committee_correct(c_test, np.array(mk)).mean(axis=1)
        q = np.percentile(accs, [5, 25, 50, 75, 95]) * 100
        size_curve[k] = {
            "min_pc": round(float(accs.min() * 100), 1),
            "p5_pc": round(float(q[0]), 1),
            "p25_pc": round(float(q[1]), 1),
            "median_pc": round(float(q[2]), 1),
            "p75_pc": round(float(q[3]), 1),
            "p95_pc": round(float(q[4]), 1),
            "max_pc": round(float(accs.max() * 100), 1),
            "prob_beat_deployable_single_82.9":
                round(float((accs > 68 / 82).mean()), 3),
            "raw": accs,
        }

    # ---- report ------------------------------------------------------------
    pool_mean = float(test_acc.mean() * 100)
    results = {
        "protocol": {
            "bootstrap_replicates": B_BOOT,
            "committee_candidates_sampled": M_COMMITTEES,
            "size_curve_samples_per_k": M_SIZE,
            "seed": SEED,
            "tie_break": "lowest index in alphabetical expert order",
        },
        "spearman_val_test_singles": round(float(
            _spearman(val_acc, test_acc)), 2),
        "pool_mean_test_acc_pc": round(pool_mean, 1),
        "policy_single": summary(dep_single),
        "policy_single_selection_concentration": {
            pool[i]: round(float((sel_single == i).mean()), 3)
            for i in np.unique(sel_single)
            if (sel_single == i).mean() >= 0.01
        },
        "policy_committee9": summary(dep_comm),
        "corr_val_test_committees9": round(float(r_comm), 2),
        "committee9_test_acc_spread_pp": round(float(
            (comm_test_acc.max() - comm_test_acc.min()) * 100), 1),
        "size_curve": {k: {kk: vv for kk, vv in v.items() if kk != "raw"}
                       for k, v in size_curve.items()},
    }

    print(json.dumps(results, indent=2))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    np.savez(OUT.with_suffix(".npz"),
             dep_single=dep_single, dep_comm=dep_comm,
             test_acc_singles=test_acc,
             comm_test_acc=comm_test_acc,
             **{f"size_k{k}": np.asarray(v["raw"])
                for k, v in size_curve.items()})
    print(f"written: {OUT}")


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


if __name__ == "__main__":
    main()
