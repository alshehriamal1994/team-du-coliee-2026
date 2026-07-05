"""Worked-example boxes for the manuscript: facts re-derived for provenance.

Three real R07 questions, chosen for what they demonstrate:
  R07-12-O  near-verbatim paraphrase of Art. 307(2); 29/30 experts correct;
            the only failure is the validation-selected champion; committee
            unanimous. Shows idiosyncratic single-model risk.
  R07-03-A  one-vote-margin rescue (5/9), pool 10/30. Shows the contested
            regime the unanimity signal flags.
  R07-28-I  Art. 670-2 exception-scope question; 1/30 experts correct
            (DeepSeek-V3.1 Standard); unwinnable by any nine-committee.
            Shows the residue committees cannot fix.

Statements verbatim from datasets/test_R07/test.jsonl; article text verbatim
from the competition civil.xml; gold from the released QA file. Writes
example_boxes_facts.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

from selection_policy_analysis import (load_gold, build_pool, DU3_EXPERTS)

HERE = Path(__file__).parent
ROOT = Path(os.environ.get("COLIEE_ROOT", "data"))
CIVIL = ROOT / "civil.xml"  # from the organisers' data package
TESTJ = ROOT / "TASK4/experiments/datasets/test_R07/test.jsonl"
OUT = HERE / "example_boxes_facts.json"

QIDS = ["R07-12-O", "R07-03-A", "R07-28-I"]


def article_text(num_jp_tag):
    text = open(CIVIL, encoding="utf-8").read()
    start = text.find(f'<Article num="{num_jp_tag}">')
    if start < 0:
        raise ValueError(num_jp_tag)
    end = text.find("</Article>", start)
    seg = text[start:end]
    body = seg.split("<text>")[1].split("</text>")[0].strip()
    return body


def main():
    gold = load_gold()
    pool, c_val, c_test, val_ids, test_ids = build_pool(gold)
    du3 = [pool.index(e) for e in DU3_EXPERTS]
    vb = pool.index("llama-3.3-70b-instruct_standard_v1")

    statements = {}
    articles_of = {}
    for line in open(TESTJ, encoding="utf-8"):
        d = json.loads(line)
        if d["id"] in QIDS:
            statements[d["id"]] = d["statement"]
            articles_of[d["id"]] = d["articles"]

    boxes = {}
    for qid in QIDS:
        i = test_ids.index(qid)
        n_pool = int(c_test[:, i].sum())
        n_comm = int(c_test[du3, i].sum())
        correct_experts = [pool[j] for j in range(30) if c_test[j, i]]
        boxes[qid] = {
            "statement_ja": statements[qid],
            "gold": gold[qid],
            "gold_articles": articles_of[qid],
            "article_text_ja": {a: article_text(a)
                                for a in articles_of[qid]},
            "pool_correct_of_30": n_pool,
            "committee_correct_of_9": n_comm,
            "committee_vote_correct": n_comm >= 5,
            "val_selected_single_correct": bool(c_test[vb, i]),
            "correct_experts_if_few": (correct_experts
                                       if n_pool <= 4 else None),
        }

    # cross-checks for the claims made in the boxes
    checks = {
        "R07-12-O_champion_only_failure":
            boxes["R07-12-O"]["pool_correct_of_30"] == 29
            and not boxes["R07-12-O"]["val_selected_single_correct"],
        "R07-12-O_committee_unanimous":
            boxes["R07-12-O"]["committee_correct_of_9"] == 9,
        "R07-03-A_one_vote_margin":
            boxes["R07-03-A"]["committee_correct_of_9"] == 5,
        "R07-28-I_single_survivor":
            boxes["R07-28-I"]["correct_experts_if_few"]
            == ["deepseek-v3.1_standard_v1"],
    }
    if not all(checks.values()):
        print("CHECKS FAILED", checks)
        sys.exit(1)
    print("box fact checks: PASS")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"boxes": boxes, "checks": checks}, f,
                  ensure_ascii=False, indent=2)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
