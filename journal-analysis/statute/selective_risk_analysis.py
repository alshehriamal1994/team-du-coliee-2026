"""Distribution-free risk control for the vote-margin abstention rule.

Section 6.5 reports the margin rule descriptively: retaining questions on which at least
some number of the nine experts agree keeps a stated coverage at a stated accuracy. Those
are point estimates on the split they are read from. A deployment question is different and
harder: if a threshold is fixed on validation data, what can be guaranteed about the error
rate among the answers the system chooses to give on data it has not seen?

This script answers that in the standard selective-prediction form. The margin threshold is
chosen on the 258 validation questions as the smallest threshold whose upper confidence
bound on selective risk (one minus accuracy among retained questions) lies at or below a
target alpha. The bound is the exact one-sided Clopper-Pearson limit at level delta, which
makes no distributional assumption and is valid for any sample size. The selected threshold
is then applied once to the 82 test questions and the realised coverage and risk reported,
so the test split is used for evaluation and never for choosing.

Two honest properties of this analysis are reported rather than hidden. First, with 82 test
questions the guarantee is loose, and the loosest useful statement is often weaker than the
descriptive point estimate it replaces; the bound is reported alongside the point estimate
so the reader sees both. Second, the guarantee is conditional on the test questions being
drawn from the same distribution as the validation ones, which the examination years make
only approximately true, so the realised test risk is reported as a check on that
assumption rather than as a further guarantee.

Verification gate: the descriptive coverage and accuracy figures recomputed here must match
margin_power_numbers.json before anything is written.

Writes selective_risk_numbers.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

import selection_policy_analysis as base

HERE = Path(__file__).parent
LEDGER = HERE / "margin_power_numbers.json"
OUT = HERE / "selective_risk_numbers.json"

DU3_MEMBERS = 9
DELTA = 0.05                     # confidence level for the bound
ALPHAS = [0.05, 0.10, 0.15]      # target selective risk


def clopper_pearson_upper(k, n, delta):
    """Exact one-sided upper confidence limit for a binomial proportion.

    k errors out of n retained. Returns the value p such that, with confidence 1-delta,
    the true error rate does not exceed p. Valid for any n, including very small n.
    """
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def margins_and_correct(pred_matrix, gold_vec):
    """Vote margin per question and whether the majority vote is correct.

    pred_matrix is (n_experts, n_questions) boolean, True where that expert is correct.
    The margin is |agree - disagree| over members, so it runs 1, 3, 5, 7, 9 for nine
    members and is read from the answers alone, needing no gold label.
    """
    n = pred_matrix.shape[0]
    n_correct_side = pred_matrix.sum(0)
    # members voting with the majority, whichever side that is
    majority_size = np.maximum(n_correct_side, n - n_correct_side)
    margin = 2 * majority_size - n
    vote_correct = n_correct_side > n / 2
    return margin, vote_correct


def curve(margin, correct):
    out = {}
    for thr in sorted(set(margin.tolist())):
        keep = margin >= thr
        n_keep = int(keep.sum())
        n_err = int((~correct[keep]).sum())
        out[int(thr)] = {
            "coverage_pc": round(100 * n_keep / len(margin), 1),
            "n_retained": n_keep,
            "n_errors": n_err,
            "selective_accuracy_pc": round(100 * (n_keep - n_err) / n_keep, 1) if n_keep else None,
            "selective_risk_pc": round(100 * n_err / n_keep, 1) if n_keep else None,
            "risk_upper_bound_pc": round(100 * clopper_pearson_upper(n_err, n_keep, DELTA), 1),
        }
    return out


def main():
    gold = base.load_gold()
    names, val, test, vids, tids = base.build_pool(gold)

    # The nine deployed members are recorded per expert in appendix_pool_numbers.json,
    # which make_appendix_pool.py generates from the same prediction logs, so the
    # membership is read rather than guessed.
    pool_led = json.loads((HERE / "appendix_pool_numbers.json").read_text())
    du3_names = [e["expert_id"] for e in pool_led["experts"] if e.get("in_du3")]
    if len(du3_names) != DU3_MEMBERS:
        raise SystemExit(f"expected {DU3_MEMBERS} DU3 members, ledger records {len(du3_names)}")
    missing = [n for n in du3_names if n not in names]
    if missing:
        raise SystemExit(f"DU3 members absent from the pool: {missing}")
    du3_idx = [names.index(n) for n in du3_names]

    mv, cv = margins_and_correct(val[du3_idx], None)
    mt, ct = margins_and_correct(test[du3_idx], None)

    led = json.loads(LEDGER.read_text())
    cur_val, cur_test = curve(mv, cv), curve(mt, ct)

    gates = {
        "val_vote_accuracy_matches_91.9": abs(100 * cv.mean() - 91.9) < 0.15,
        "test_vote_accuracy_matches_93.9": abs(100 * ct.mean() - 93.9) < 0.15,
        "val_unanimous_share_matches_52.7":
            abs(cur_val[9]["coverage_pc"] - led["margin_val"]["coverage"]["margin>=9"]["coverage_pc"]) < 0.15,
        "test_unanimous_share_matches_73.2":
            abs(cur_test[9]["coverage_pc"] - led["margin_test_R07"]["coverage"]["margin>=9"]["coverage_pc"]) < 0.15,
    }
    if not all(gates.values()):
        raise SystemExit(f"GATE FAILED, nothing written: {gates}\n"
                         f"val acc {100*cv.mean():.2f}, test acc {100*ct.mean():.2f}, "
                         f"val cov@9 {cur_val[9]['coverage_pc']}, test cov@9 {cur_test[9]['coverage_pc']}")

    calibrated = {}
    for a in ALPHAS:
        chosen = None
        for thr in sorted(cur_val):                      # smallest threshold meeting the bound
            if cur_val[thr]["risk_upper_bound_pc"] <= 100 * a:
                chosen = thr
                break
        if chosen is None:
            calibrated[f"alpha_{a}"] = {"threshold": None,
                                        "note": "no threshold meets this target on validation"}
            continue
        calibrated[f"alpha_{a}"] = {
            "threshold_selected_on_validation": chosen,
            "validation": cur_val[chosen],
            "test_realised": cur_test[chosen],
            "guarantee": (f"with 95% confidence, selective risk at margin >= {chosen} is at most "
                          f"{cur_val[chosen]['risk_upper_bound_pc']}% on data distributed as the "
                          f"validation split"),
        }

    res = {
        "purpose": "distribution-free risk control for the vote-margin abstention rule",
        "method": ("threshold chosen on 258 validation questions as the smallest whose exact "
                   "one-sided Clopper-Pearson upper bound on selective risk meets the target; "
                   "applied once to the 82 test questions"),
        "delta": DELTA,
        "gates": {k: bool(v) for k, v in gates.items()},
        "validation_curve": cur_val,
        "test_curve": cur_test,
        "calibrated": calibrated,
        "reading": {
            "bounds_are_loose_at_this_size": (
                "The 82-question test split cannot support a tight guarantee. Where the bound "
                "is weaker than the descriptive point estimate, both are reported and the "
                "bound is the claim."),
            "distribution_assumption": (
                "Validity requires the test questions to be exchangeable with the validation "
                "ones. They come from a later examination year, so the realised test risk is "
                "reported as a check on that assumption rather than as a second guarantee."),
        },
    }
    OUT.write_text(json.dumps(res, indent=2))

    print("  validation curve (threshold: coverage, risk, 95% upper bound)")
    for thr, d in cur_val.items():
        print(f"    m>={thr}: cov {d['coverage_pc']:5.1f}%  risk {d['selective_risk_pc']:5.1f}%  "
              f"bound {d['risk_upper_bound_pc']:5.1f}%")
    print("\n  calibrated thresholds")
    for k, v in calibrated.items():
        if v.get("threshold_selected_on_validation") is None:
            print(f"    {k}: none met"); continue
        t = v["test_realised"]
        print(f"    {k}: margin >= {v['threshold_selected_on_validation']}  "
              f"| val bound {v['validation']['risk_upper_bound_pc']}%  "
              f"| test coverage {t['coverage_pc']}% risk {t['selective_risk_pc']}%")
    print(f"\nwritten: {OUT.name}")


if __name__ == "__main__":
    main()
