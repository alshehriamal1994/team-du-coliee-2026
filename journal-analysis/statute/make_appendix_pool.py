"""Generates the complete expert-pool appendix (journal article, Appendix).

Lists every configuration in the thirty-expert pool with its base model, parameter count,
architecture, prompt, release month, availability at submission time and accuracy on the
258-question validation benchmark. Identities and accuracies are recomputed from the same
prediction logs used throughout the article, via selection_policy_analysis.py, so the
table cannot drift from the analysis. Release months were checked against vendor
announcements and model cards, and month precision is deliberate, being what availability
relative to the submission date requires.

Writes appendix_pool.tex and appendix_pool_numbers.json.
"""

import json
from pathlib import Path

import selection_policy_analysis as base

HERE = Path(__file__).resolve().parent
TEX = HERE / "appendix_pool.tex"
LEDGER = HERE / "appendix_pool_numbers.json"

# base model -> (display name, parameters, architecture, release month)
MODELS = {
    "deepseek-r1": ("DeepSeek-R1", "671B", "MoE", "Jan 2025"),
    "deepseek-v3.1": ("DeepSeek-V3.1", "671B", "MoE", "Aug 2025"),
    "llama-3.3-70b-instruct": ("Llama-3.3-70B", "70B", "dense", "Dec 2024"),
    "llama-4-maverick": ("Llama-4-Maverick", "400B", "MoE", "Apr 2025"),
    "llama-4-scout": ("Llama-4-Scout", "109B", "MoE", "Apr 2025"),
    "qwen-2.5-72b-instruct": ("Qwen2.5-72B", "72B", "dense", "Sep 2024"),
    "qwen3-235b-a22b": ("Qwen3-235B-A22B", "235B", "MoE", "Apr 2025"),
    "qwen3-32b": ("Qwen3-32B", "32B", "dense", "Apr 2025"),
    "qwq-32b": ("QwQ-32B", "32B", "dense", "Mar 2025"),
    "mistral-large-2411": ("Mistral-Large-2411", "123B", "dense", "Nov 2024"),
    "gemma-3-27b-it": ("Gemma-3-27B", "27B", "dense", "Mar 2025"),
}

PROMPTS = {
    "standard": "Standard", "concise": "Concise", "cot_strict": "CoT (strict)",
    "irac": "IRAC", "sc2": "Self-consistency ($k{=}2$)",
    "sc3": "Self-consistency ($k{=}3$)", "meticulous": "Meticulous",
    "generous": "Generous", "english": "English", "zs": "Zero-shot",
}

# configurations whose logs completed only after the competition
POST_COMPETITION_PREFIXES = ("deepseek-v3.1",)


def parse(name):
    """expert id -> (base model key, prompt key, variant suffix)."""
    if name == "deepseek-r1-zs":
        return "deepseek-r1", "zs", ""
    stem = name
    variant = ""
    for suf in ("_v1", "_v2", "_v3"):
        if stem.endswith(suf):
            stem, variant = stem[: -len(suf)], suf[1:]
            break
    for key in sorted(MODELS, key=len, reverse=True):
        if stem.startswith(key + "_"):
            return key, stem[len(key) + 1:], variant
    raise ValueError(f"unparsed expert id: {name}")


def main():
    gold = base.load_gold()
    pool, c_val, c_test, _, _ = base.build_pool(gold)
    val_acc = c_val.mean(axis=1) * 100
    du3 = set(base.DU3_EXPERTS)

    rows = []
    for i, name in enumerate(pool):
        mk, pk, variant = parse(name)
        disp, params, arch, released = MODELS[mk]
        if pk not in PROMPTS:
            raise ValueError(f"unknown prompt {pk!r} in {name}")
        available = not name.startswith(POST_COMPETITION_PREFIXES)
        rows.append({
            "expert_id": name, "model": disp, "params": params, "arch": arch,
            "prompt": PROMPTS[pk], "variant": variant, "released": released,
            "available_at_submission": available,
            "in_du3": name in du3,
            "val_acc_pc": round(float(val_acc[i]), 1),
        })

    rows.sort(key=lambda r: (-r["val_acc_pc"], r["model"], r["prompt"]))
    n_avail = sum(r["available_at_submission"] for r in rows)
    n_du3 = sum(r["in_du3"] for r in rows)
    assert len(rows) == 30 and n_du3 == 9, (len(rows), n_du3)

    lines = [
        r"\section{The complete expert pool}\label{app:pool}",
        r"Table~\ref{tab:pool} lists every configuration in the thirty-expert "
        r"pool analysed in Section~\ref{sec:selection}: the base model, its "
        r"parameter count and architecture, the prompt, the release month of "
        r"the base model, whether the configuration was in the search space at "
        r"submission time, and its accuracy on the 258-question validation "
        r"benchmark. Accuracies are recomputed from the released prediction "
        r"logs. The nine members of the submitted plain vote (DU3) are marked. "
        r"Release dates are given to the month, which is the precision "
        r"availability relative to submission requires, and avoids asserting "
        r"exact days that differ between the vendor announcement, the model "
        r"card and the weights upload.",
        "",
        r"\begin{table}[t]",
        r"\caption{The complete thirty-expert pool. ``Sub.'' marks the "
        r"configurations available when the runs were submitted; the two "
        r"DeepSeek-V3.1 configurations completed only afterwards and appear in "
        r"the post-competition analysis alone. ``DU3'' marks the nine members "
        r"of the submitted plain vote. Accuracy (\%) on the 258-question "
        r"validation benchmark. Ledger: "
        r"\texttt{appendix\_pool\_numbers.json}.}\label{tab:pool}",
        r"\begin{tabular}{@{}rlllllcc r@{}}",
        r"\toprule",
        r"\# & Base model & Par. & Arch. & Prompt & Released & Sub. & DU3 & Acc. \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i} & {r['model']} & {r['params']} & {r['arch']} & "
            f"{r['prompt']} & {r['released']} & "
            f"{'\\checkmark' if r['available_at_submission'] else '--'} & "
            f"{'\\checkmark' if r['in_du3'] else '--'} & "
            f"{r['val_acc_pc']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    TEX.write_text("\n".join(lines) + "\n")

    ledger = {
        "purpose": "complete expert pool listing for app:pool",
        "n_experts": len(rows),
        "n_available_at_submission": n_avail,
        "n_in_du3": n_du3,
        "n_base_models": len({r["model"] for r in rows}),
        "n_prompts": len({r["prompt"] for r in rows}),
        "val_acc_range_pc": [min(r["val_acc_pc"] for r in rows),
                             max(r["val_acc_pc"] for r in rows)],
        "release_dates_verified": "2026-08-05, vendor announcements and model "
                                  "cards, month precision",
        "experts": rows,
    }
    LEDGER.write_text(json.dumps(ledger, indent=2))
    print(f"{len(rows)} experts, {n_avail} available at submission, "
          f"{n_du3} in DU3, {ledger['n_base_models']} base models, "
          f"{ledger['n_prompts']} prompts, "
          f"val {ledger['val_acc_range_pc'][0]}-{ledger['val_acc_range_pc'][1]}%")
    print(f"written: {TEX.name}, {LEDGER.name}")


if __name__ == "__main__":
    main()
