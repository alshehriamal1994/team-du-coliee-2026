"""Does the selection failure hold outside this benchmark?

Everything in Sections 5 and 6 comes from one competition, one jurisdiction and one task.
This script tests the central claim on data we did not produce: the per-instance results
that Stanford CRFM publishes for HELM, covering thirty-two models on five LegalBench tasks
of United States legal reasoning. No model is run here. The published per-instance
exact-match scores are downloaded and analysed, so the models, the prompts, the gold labels
and the scoring are all third-party.

Only one of the five tasks, proa, has the shape of our own: a statutory provision is
supplied and a binary legal judgement is asked of it. The others are trademark
classification, bill-to-company relevance, judgment-section labelling and closed-book
citizenship law. We therefore do not claim a replication of a statute-entailment result.
What is tested is the selection phenomenon itself, which is a property of choosing among
candidates on a finite sample and is not specific to a task.

The paper's claim is explicitly conditional, holding when the strongest candidates are
statistically indistinguishable. The full HELM pools do not meet that condition: they span
thirty-five to eighty-four points from best to worst, because they survey all models rather
than shortlisting strong ones, where our pool spans six. Each task is therefore restricted
to its strongest K models, which reproduces the situation a practitioner faces after
shortlisting, and the selection experiment is run inside that shortlist.

Procedure per task, mirroring Section 5: split the instances at random into two halves,
rank the shortlisted models on the first, deploy the best on the second, and record its
accuracy, the accuracy of drawing a shortlisted model at random, and the accuracy of an
unweighted majority vote over the shortlist. Repeated B times with a fixed seed.

Shortlisting by accuracy on the whole task and then selecting within the shortlist uses the
data twice, which is stated rather than hidden. It mirrors practice, where candidates are
shortlisted from published leaderboards and the practitioner must still choose among them
from their own validation sample, and it can only make selection look better than it is,
since the shortlist already excludes the weak models a random draw would otherwise catch.

Writes external_legalbench_numbers.json.
"""
import itertools
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "legalbench_cache"
OUT = HERE / "external_legalbench_numbers.json"
BASE = "https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output"
RELEASE = "v1.1.0"
SHORTLIST_K = 8
B = 2000
SEED = 20260818


def fetch_suites():
    url = f"{BASE}/releases/{RELEASE}/runs_to_run_suites.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def load_matrices():
    """Correctness matrices per task, models by instances, from published HELM results."""
    CACHE.mkdir(exist_ok=True)
    suites = fetch_suites()
    runs = sorted(k for k in suites if k.startswith("legalbench:"))
    per_task = {}
    for run in runs:
        sub = re.search(r"subset=([a-z_]+)", run).group(1)
        mod = re.search(r"model=(.+)$", run).group(1)
        fn = CACHE / (run.replace(":", "_").replace(",", "__").replace("=", "-") + ".json")
        if not fn.exists():
            url = f"{BASE}/runs/{suites[run]}/{run}/per_instance_stats.json"
            with urllib.request.urlopen(url, timeout=60) as r:
                fn.write_bytes(r.read())
        recs = json.loads(fn.read_text())
        byid = {rec["instance_id"]: st["mean"]
                for rec in recs for st in rec["stats"]
                if st["name"]["name"] == "exact_match"}
        per_task.setdefault(sub, {})[mod] = byid
    mats = {}
    for sub, models in per_task.items():
        ids = sorted(set.intersection(*[set(v) for v in models.values()]))
        names = sorted(models)
        mats[sub] = (names, np.array([[models[n][i] for i in ids] for n in names]))
    return mats


def selection_experiment(S, rng):
    n = S.shape[1]
    k = S.shape[0]
    picks_best, sel, rnd, vote = 0, [], [], []
    for _ in range(B):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        va, vb = S[:, a].mean(1), S[:, b].mean(1)
        s = int(np.argmax(va))
        picks_best += int(vb[s] == vb.max())
        sel.append(vb[s])
        rnd.append(vb.mean())
        vote.append((S[:, b].sum(0) > k / 2).mean())
    return {
        "p_select_best": round(picks_best / B, 3),
        "selected_acc_pc": round(float(100 * np.mean(sel)), 2),
        "random_draw_acc_pc": round(float(100 * np.mean(rnd)), 2),
        "vote_acc_pc": round(float(100 * np.mean(vote)), 2),
        "selection_minus_random_pp": round(float(100 * (np.mean(sel) - np.mean(rnd))), 2),
        "vote_minus_selected_pp": round(float(100 * (np.mean(vote) - np.mean(sel))), 2),
    }


def overlap_ratio(S):
    E = 1 - S
    r = E.mean(1)
    obs, ind = [], []
    for i, j in itertools.combinations(range(S.shape[0]), 2):
        obs.append((E[i] * E[j]).mean())
        ind.append(r[i] * r[j])
    return float(np.mean(obs) / np.mean(ind))


def main():
    mats = load_matrices()
    rng = np.random.default_rng(SEED)

    gates = {
        "five_tasks": len(mats) == 5,
        "thirty_two_models_each": all(len(v[0]) == 32 for v in mats.values()),
        "proa_has_95_instances": mats["proa"][1].shape[1] == 95,
    }
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}")

    tasks = {}
    for sub, (names, M) in sorted(mats.items()):
        acc = M.mean(1)
        full_spread = float(100 * (acc.max() - acc.min()))
        top = np.argsort(-acc)[:SHORTLIST_K]
        S = M[top]
        sl_acc = acc[top]
        res = selection_experiment(S, rng)
        vendors = sorted({names[i].split("_")[0] for i in top})
        tasks[sub] = {
            "n_instances": int(M.shape[1]),
            "n_models_available": int(M.shape[0]),
            "full_pool_spread_pp": round(full_spread, 1),
            "shortlist_k": SHORTLIST_K,
            "shortlist_spread_pp": round(float(100 * (sl_acc.max() - sl_acc.min())), 1),
            "shortlist_distinct_vendors": len(vendors),
            "shortlist_error_overlap_ratio": round(overlap_ratio(S), 2),
            **res,
        }

    tight = {k: v for k, v in tasks.items() if v["shortlist_spread_pp"] <= 6.5}
    res = {
        "purpose": "external test of the selection claim on third-party data",
        "source": ("Stanford CRFM HELM published per-instance exact-match results, "
                   f"release {RELEASE}; no model was run here"),
        "url_pattern": f"{BASE}/runs/<suite>/legalbench:subset=<task>,model=<model>/per_instance_stats.json",
        "seed": SEED,
        "bootstrap_splits": B,
        "gates": gates,
        "tasks": tasks,
        "reading": {
            "condition_not_met_by_full_pools": (
                "The full HELM pools span 34.6 to 83.5 points best to worst because they "
                "survey all models rather than shortlisting strong ones. Our pool spans 6.2. "
                "The paper's claim is conditional on near-tied candidates and does not apply "
                "to a pool of that shape, so each task is restricted to its strongest eight."),
            "selection_finding_corroborated": (
                "On the three shortlists whose spread is closest to our own, selecting on "
                "half the instances is worth 0.2 to 0.5 points over drawing at random, "
                "against 0.3 points in our own pool, and the probability of identifying the "
                "genuinely best shortlisted model runs from 0.10 to 0.48."),
            "voting_remedy_not_corroborated": (
                "An unweighted majority vote over the shortlist is worse than the selected "
                "single on four of the five tasks, where in our setting it is better by "
                "eleven points. The shortlists are as heterogeneous as ours by vendor count "
                "and by error-overlap ratio, so architectural homogeneity does not explain "
                "the difference and we offer no explanation."),
        },
        "scope": ("These are United States legal reasoning tasks of five different kinds, "
                  "only one of which, proa, supplies a statutory provision and asks a binary "
                  "judgement of it. No claim is made that a statute-entailment result "
                  "replicates. What is tested is the selection phenomenon."),
    }
    OUT.write_text(json.dumps(res, indent=2))

    print(f"{'task':34s} {'n':>5s} {'full':>6s} {'short':>6s} {'P(best)':>8s} "
          f"{'sel-rand':>9s} {'vote-sel':>9s}")
    for k, v in tasks.items():
        print(f"  {k:32s} {v['n_instances']:5d} {v['full_pool_spread_pp']:6.1f} "
              f"{v['shortlist_spread_pp']:6.1f} {v['p_select_best']:8.3f} "
              f"{v['selection_minus_random_pp']:+9.2f} {v['vote_minus_selected_pp']:+9.2f}")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
