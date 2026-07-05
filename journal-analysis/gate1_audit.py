"""Gate-1 audit for Manuscript 1 (statute, Tasks 3+4).

Four jobs, all from frozen artefacts on disk (no new inference):
  A. R07 gold provenance: cross-check the three independent gold sources.
  B. Score the FROZEN official submission files task4-{H30,R01,R02,TE}.DU{1,2,3}
     against gold, and verify the reproduced 9-expert vote (DU3) matches the
     frozen DU3 file line-for-line.
  C. McNemar exact tests on the R07 accuracy ladder
     (deployable val-selected single vs DU3 vote vs official DU1).
  D. Noise-only null simulation: with each expert's true skill fixed at its
     validation accuracy, how much val->test rank instability does an
     82-question test produce by sampling noise alone?  Compares observed
     Pearson r, selection regret, and val-best test rank against the null.

Gate: reproduces the ledger numbers in deep_analysis_numbers.json
(pool of 30, val-best = llama-3.3-70b-instruct_standard_v1 at 68/82,
DU3 vote 77/82, r = 0.43) and aborts if any of them fails to reproduce.

Outputs: gate1_audit_numbers.json (+ stdout summary).
Seed 20260704 to match selection_policy_analysis.py.
"""

import json
import os
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))
RUNS = ROOT / "TASK4/experiments/runs_ensemble"
DATASETS = ROOT / "TASK4/experiments/datasets"
FROZEN = ROOT / "TASK4/COLIEE2026_Official_Submission"
POSTCOMP = ROOT / "TASK33/Post_Competition_Analysis_Task4_Ensemble_on_Task3"
OUT = Path(__file__).parent / "gate1_audit_numbers.json"

VAL_SPLITS = ["H30", "R01", "R02"]
SEED = 20260704
B = 10_000

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

# Pool rule (pre-registered): base-LLM API configs with prediction files on all
# four splits; excludes meta-level outputs and fine-tuned encoders.
EXCLUDE = re.compile(
    r"stacking|neural|deliberation|deberta|ce-base|mbert|bert-large|modernbert"
    r"|qwen7|qwen-zs|llama33-zs|oldlaw|fewshot|thinking"
)


def read_preds(path):
    preds = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            preds[parts[0]] = parts[1]
    return preds


def load_gold():
    gold = {}
    for sp in VAL_SPLITS:
        with open(DATASETS / f"{sp}_formal/test.jsonl") as f:
            gold[sp] = {j["id"]: j["label"] for j in map(json.loads, f)}
    # R07: three sources.
    mv = [json.loads(l) for l in open(POSTCOMP / "ensemble_majority_vote.jsonl")]
    r07_mv = {j["id"]: j["gold"] for j in mv}
    av = [json.loads(l) for l in open(POSTCOMP / "adversarial_verification.jsonl")]
    r07_av = {j["id"]: j["gold"] for j in av if j.get("gold")}
    with open(DATASETS / "test_R07/test.jsonl") as f:
        r07_ds = {j["id"]: j["label"] for j in map(json.loads, f)}
    ds_labels = sorted({v for v in r07_ds.values()})
    av_clash = {q for q, g in r07_av.items() if r07_mv.get(q) != g}
    provenance = {
        "n_R07_gold_mv": len(r07_mv),
        "n_overlap_adversarial": len(r07_av),
        "adversarial_disagreements": sorted(av_clash),
        "test_R07_dataset_label_values": ds_labels,
        "note": (
            "primary gold = ensemble_majority_vote.jsonl; adversarial_verification.jsonl "
            "agrees on all overlapping ids" if not av_clash else "DISAGREEMENT - inspect"
        ),
    }
    gold["R07"] = r07_mv
    return gold, provenance


def acc(preds, gold):
    hits = [preds[q] == g for q, g in gold.items() if q in preds]
    assert len(hits) == len(gold), f"missing questions: {len(hits)} vs {len(gold)}"
    return sum(hits), len(hits)


def majority(expert_preds, qids):
    out = {}
    for q in qids:
        votes = [p[q] for p in expert_preds if q in p]
        assert len(votes) == len(expert_preds), f"missing vote on {q}"
        out[q] = "Y" if votes.count("Y") > votes.count("N") else "N"
    return out


def mcnemar_exact(pred_a, pred_b, gold):
    b = c = 0  # b: A right / B wrong ; c: A wrong / B right
    for q, g in gold.items():
        ra, rb = pred_a[q] == g, pred_b[q] == g
        if ra and not rb:
            b += 1
        elif rb and not ra:
            c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return b, c, min(1.0, 2 * tail)


def main():
    rng = np.random.default_rng(SEED)
    gold, provenance = load_gold()
    report = {"seed": SEED, "R07_gold_provenance": provenance}

    # --- B1. frozen official runs -------------------------------------------
    frozen = {}
    for run in ["DU1", "DU2", "DU3"]:
        per = {}
        for sp, tag in [("H30", "H30"), ("R01", "R01"), ("R02", "R02"), ("R07", "TE")]:
            h, n = acc(read_preds(FROZEN / f"task4-{tag}.{run}"), gold[sp])
            per[sp] = f"{h}/{n} = {100*h/n:.1f}%"
        vh = sum(acc(read_preds(FROZEN / f"task4-{s}.{run}"), gold[s])[0] for s in VAL_SPLITS)
        per["val_258"] = f"{vh}/258 = {100*vh/258:.1f}%"
        frozen[run] = per
    report["frozen_official_runs"] = frozen

    # --- B2. reproduce DU3 from the 9 expert files --------------------------
    du3_repro = {}
    du3_preds_r07 = None
    for sp in VAL_SPLITS + ["R07"]:
        experts = [read_preds(RUNS / f"{sp}.{e}") for e in DU3_EXPERTS]
        vote = majority(experts, gold[sp].keys())
        h, n = acc(vote, gold[sp])
        official = read_preds(FROZEN / f"task4-{'TE' if sp == 'R07' else sp}.DU3")
        agree = sum(vote[q] == official[q] for q in gold[sp])
        du3_repro[sp] = {
            "reproduced_vote": f"{h}/{n} = {100*h/n:.1f}%",
            "agreement_with_frozen_DU3": f"{agree}/{n}",
        }
        if sp == "R07":
            du3_preds_r07 = vote
    report["DU3_reproduction"] = du3_repro

    # --- pool of 30 ----------------------------------------------------------
    names = sorted(
        {
            p.name.split(".", 1)[1]
            for p in RUNS.iterdir()
            if not EXCLUDE.search(p.name)
        }
    )
    pool = [
        e
        for e in names
        if all((RUNS / f"{sp}.{e}").exists() for sp in VAL_SPLITS + ["R07"])
    ]
    val_gold = {q: g for sp in VAL_SPLITS for q, g in gold[sp].items()}
    val_acc, test_acc, test_hits = {}, {}, {}
    for e in pool:
        vp = {}
        for sp in VAL_SPLITS:
            vp.update(read_preds(RUNS / f"{sp}.{e}"))
        h, n = acc(vp, val_gold)
        val_acc[e] = h / n
        tp = read_preds(RUNS / f"R07.{e}")
        h, n = acc(tp, gold["R07"])
        test_acc[e] = h / n
        test_hits[e] = tp
    va = np.array([val_acc[e] for e in pool])
    ta = np.array([test_acc[e] for e in pool])
    r_obs = float(np.corrcoef(va, ta)[0, 1])
    best_val = pool[int(np.argmax(va))]
    n_higher = int((ta > test_acc[best_val]).sum())
    n_tied = int((ta == test_acc[best_val]).sum() - 1)
    regret_obs = float(ta.max() - test_acc[best_val])

    # --- ledger gate ---------------------------------------------------------
    assert len(pool) == 30, f"pool size {len(pool)} != 30: {pool}"
    assert best_val == "llama-3.3-70b-instruct_standard_v1", best_val
    assert round(test_acc[best_val] * 82) == 68, test_acc[best_val]
    assert du3_repro["R07"]["reproduced_vote"].startswith("77/82")
    assert abs(r_obs - 0.43) < 0.01, r_obs
    report["ledger_gate"] = (
        "PASS: pool=30, val-best=llama-3.3-70b standard 68/82, DU3 vote 77/82, "
        f"r={r_obs:.2f} - matches deep_analysis_numbers.json"
    )
    report["pool"] = {
        "rule": "base-LLM API configs with files on all 4 splits; excludes "
        "meta outputs (deliberation/stacking/neural fusion), fine-tuned "
        "encoders, and single-split variants (oldlaw/fewshot/thinking)",
        "n": len(pool),
        "val_best": best_val,
        "val_best_test_rank": f"{n_higher + 1} of 30 ({n_higher} strictly higher, "
        f"{n_tied} tied)",
        "selection_regret_pp": round(100 * regret_obs, 1),
        "pearson_r_val_test": round(r_obs, 3),
    }

    # --- C. McNemar ladder ---------------------------------------------------
    deploy = test_hits[best_val]
    du1 = read_preds(FROZEN / "task4-TE.DU1")
    g = gold["R07"]
    ladder = {}
    for label, a, b_ in [
        ("deployable_single_vs_DU3_vote", deploy, du3_preds_r07),
        ("deployable_single_vs_official_DU1", deploy, du1),
        ("DU3_vote_vs_official_DU1", du3_preds_r07, du1),
    ]:
        bb, cc, p = mcnemar_exact(a, b_, g)
        ladder[label] = {"first_only_correct": bb, "second_only_correct": cc,
                         "exact_two_sided_p": round(p, 5)}
    report["mcnemar_R07"] = ladder

    # --- D. noise-only null --------------------------------------------------
    # Null: each expert's true skill = its validation accuracy; test outcomes
    # are Binomial(82, skill). Sampling noise only - no distribution shift.
    sims_r = np.empty(B)
    sims_regret = np.empty(B)
    sims_rank = np.empty(B, dtype=int)
    i_best = int(np.argmax(va))
    for i in range(B):
        t = rng.binomial(82, va) / 82
        sims_r[i] = np.corrcoef(va, t)[0, 1]
        sims_regret[i] = t.max() - t[i_best]
        sims_rank[i] = int((t > t[i_best]).sum()) + 1
    report["noise_only_null"] = {
        "B": B,
        "observed": {
            "pearson_r": round(r_obs, 3),
            "regret_pp": round(100 * regret_obs, 1),
            "val_best_rank": n_higher + 1,
        },
        "null_mean": {
            "pearson_r": round(float(sims_r.mean()), 3),
            "regret_pp": round(float(100 * sims_regret.mean()), 1),
            "val_best_rank": round(float(sims_rank.mean()), 1),
        },
        "null_90pct_interval": {
            "pearson_r": [round(float(np.quantile(sims_r, q)), 3) for q in (0.05, 0.95)],
            "regret_pp": [round(float(100 * np.quantile(sims_regret, q)), 1) for q in (0.05, 0.95)],
            "val_best_rank": [int(np.quantile(sims_rank, q)) for q in (0.05, 0.95)],
        },
        "p_values_one_sided": {
            "P(r_null <= r_obs)": round(float((sims_r <= r_obs).mean()), 4),
            "P(regret_null >= regret_obs)": round(float((sims_regret >= regret_obs).mean()), 4),
            "P(rank_null >= rank_obs)": round(float((sims_rank >= n_higher + 1).mean()), 4),
        },
        "caveat": (
            "The null fixes each expert's true skill at its validation accuracy, "
            "so validation-side estimation noise (winner's curse on the val-best) "
            "is not modelled; the rank p-value may partly reflect that rather than "
            "distribution shift. Safe wording: the observed val-test correlation is "
            "consistent with sampling noise alone - an 82-question test cannot rank "
            "this pool, so neither can a 258-question validation set; selection is "
            "unreliable whatever the mechanism."
        ),
    }

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
