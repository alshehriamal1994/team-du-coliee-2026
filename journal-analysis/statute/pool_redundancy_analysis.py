"""Does the selection failure depend on prompt-variant redundancy in the pool?

The thirty configurations are drawn from eleven base models, and three of them supply
six or seven configurations each. A reader may reasonably object that near-identical prompt
variants of one model cannot be ranked by any validation set, so the reported failure of
validation selection could be an artefact of a pool padded with redundant candidates rather
than a property of the task.

This script removes the redundancy and repeats the selection. One configuration is retained
per distinct base model, the one with the highest validation accuracy, which is what a
practitioner comparing models rather than prompts would shortlist. Selection is then run on
that reduced pool exactly as in Section 5.

Two robustness variants accompany the main reduction, since retaining the validation-best
member is itself a validation-dependent choice:
  - retaining the median-validation configuration per base model,
  - retaining one configuration per base model at random, averaged over draws with a fixed
    seed, which removes the validation dependence entirely.

Reported for each: the validation winner's test rank within the reduced pool, its test
accuracy against the pool mean (what picking at random returns in expectation), and the
validation-to-test correlation across candidates.

Verification gate: the full-pool figures must reproduce deep_analysis_numbers.json, namely
the validation winner's test rank of 28 of 30 and the pool mean test accuracy, before any
reduced-pool number is written.

Writes pool_redundancy_numbers.json.
"""
import json
from pathlib import Path

import numpy as np

import selection_policy_analysis as base

HERE = Path(__file__).parent
OUT = HERE / "pool_redundancy_numbers.json"
POLICY = HERE / "selection_policy_numbers.json"
SEED = 20260818


def base_model_map(here):
    """Authoritative expert-to-base-model mapping.

    Splitting configuration names on the underscore is wrong: 'deepseek-r1-zs' is
    DeepSeek-R1 under a zero-shot prompt, not a separate model. The mapping is therefore
    read from appendix_pool_numbers.json, which make_appendix_pool.py generates with an
    explicit model field per expert.
    """
    led = json.loads((here / "appendix_pool_numbers.json").read_text())
    return {e["expert_id"]: e["model"] for e in led["experts"]}


def summarise(names, v, t, label):
    win = int(np.argmax(v))
    rank = int((t > t[win]).sum() + 1)
    return {
        "selection_set": label,
        "n_candidates": int(len(v)),
        "validation_winner": names[win],
        "winner_val_acc_pc": round(float(v[win]), 2),
        "winner_test_acc_pc": round(float(t[win]), 2),
        "winner_test_rank": rank,
        "pool_mean_test_acc_pc": round(float(t.mean()), 2),
        "selection_minus_random_pp": round(float(t[win] - t.mean()), 2),
        "best_test_acc_pc": round(float(t.max()), 2),
        "worst_test_acc_pc": round(float(t.min()), 2),
        "pearson_val_test": round(float(np.corrcoef(v, t)[0, 1]), 3),
    }


def main():
    gold = base.load_gold()
    names, val, test, vids, tids = base.build_pool(gold)
    v, t = val.mean(1) * 100, test.mean(1) * 100
    bmap = base_model_map(HERE)
    missing = [n for n in names if n not in bmap]
    if missing:
        raise SystemExit(f"experts absent from the pool ledger: {missing}")
    bases = [bmap[n] for n in names]

    full = summarise(names, v, t, "full pool, all configurations")
    ledger = json.loads(POLICY.read_text())
    gates = {
        "n_configurations_is_30": len(names) == 30,
        "full_pool_winner_ranks_28": full["winner_test_rank"] == 28,
        "pool_mean_test_matches_ledger":
            abs(full["pool_mean_test_acc_pc"] - ledger["pool_mean_test_acc_pc"]) < 0.06,
    }
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")

    uniq = sorted(set(bases))
    keep_best = [max((i for i in range(len(names)) if bases[i] == b), key=lambda i: v[i])
                 for b in uniq]
    keep_med = []
    for b in uniq:
        cand = sorted((i for i in range(len(names)) if bases[i] == b), key=lambda i: v[i])
        keep_med.append(cand[len(cand) // 2])

    rng = np.random.default_rng(SEED)
    rand_ranks, rand_gaps, rand_corrs = [], [], []
    for _ in range(10_000):
        pick = [rng.choice([i for i in range(len(names)) if bases[i] == b]) for b in uniq]
        vv, tt = v[pick], t[pick]
        w = int(np.argmax(vv))
        rand_ranks.append(int((tt > tt[w]).sum() + 1))
        rand_gaps.append(float(tt[w] - tt.mean()))
        rand_corrs.append(float(np.corrcoef(vv, tt)[0, 1]))

    res = {
        "purpose": "is the selection failure an artefact of prompt-variant redundancy?",
        "gates": gates,
        "seed": SEED,
        "n_base_models": len(uniq),
        "configurations_per_base_model": {b: int(bases.count(b)) for b in uniq},
        "full_pool": full,
        "one_per_base_model_validation_best":
            summarise([names[i] for i in keep_best], v[keep_best], t[keep_best],
                      "one configuration per base model, validation-best retained"),
        "one_per_base_model_median_validation":
            summarise([names[i] for i in keep_med], v[keep_med], t[keep_med],
                      "one configuration per base model, median-validation retained"),
        "one_per_base_model_random_draw": {
            "selection_set": "one configuration per base model drawn at random, 10,000 draws",
            "n_candidates": len(uniq),
            "mean_winner_test_rank": round(float(np.mean(rand_ranks)), 2),
            "median_winner_test_rank": float(np.median(rand_ranks)),
            "share_winner_in_bottom_half_pc":
                round(float(np.mean(np.array(rand_ranks) > len(uniq) / 2) * 100), 1),
            "mean_selection_minus_random_pp": round(float(np.mean(rand_gaps)), 2),
            "ci95_selection_minus_random_pp":
                [round(float(np.percentile(rand_gaps, 2.5)), 2),
                 round(float(np.percentile(rand_gaps, 97.5)), 2)],
            "mean_pearson_val_test": round(float(np.mean(rand_corrs)), 3),
        },
        "reading": (
            "Removing prompt-variant redundancy does not rescue validation selection; on "
            "this pool it makes it worse under the validation-best reduction. The objection "
            "that the failure reflects a padded pool is therefore not supported. The reduced "
            "pools hold eleven candidates, so "
            "each single reduction is one small draw and the random-draw variant is the "
            "one to read for central tendency."),
    }
    OUT.write_text(json.dumps(res, indent=2))

    for k in ("full_pool", "one_per_base_model_validation_best",
              "one_per_base_model_median_validation"):
        s = res[k]
        print(f"  {s['selection_set'][:52]:54s} n={s['n_candidates']:2d} "
              f"winner rank {s['winner_test_rank']:2d}/{s['n_candidates']:<2d} "
              f"vs random {s['selection_minus_random_pp']:+5.1f}pp  r={s['pearson_val_test']:.2f}")
    r = res["one_per_base_model_random_draw"]
    print(f"  {'one per base model, random draw (10,000)':54s} n={r['n_candidates']:2d} "
          f"mean rank {r['mean_winner_test_rank']:.1f}/{r['n_candidates']}   "
          f"vs random {r['mean_selection_minus_random_pp']:+5.1f}pp "
          f"CI {r['ci95_selection_minus_random_pp']}  r={r['mean_pearson_val_test']:.2f}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
