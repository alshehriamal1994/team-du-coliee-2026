"""How often does a Task 1 gold citation carry negative treatment?

The Task 1 label records the cases a court referred to, not the cases that
support the propositions in the query. Courts cite in order to distinguish,
limit or doubt as well as to follow, so a returned set is a claim about the
material the court engaged with rather than about the strength of a proposition.
This script measures how often that distinction is visible in the collection, by
counting query cases in which a citation-suppression marker stands close to
language of distinguishing or negative treatment.

The measure is a lower bound on prevalence and an upper bound on nothing: it
detects only treatment signalled by an explicit verb near the marker, and the
window is characters rather than sentences. It is offered to bound the reading
the article places on answer-set size, not to classify citations.

Gated on finding the expected 400 query cases and the expected gold total.

Writes expY_numbers.json.
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = os.environ.get("COLIEE_ROOT", "data")
HERE = Path(__file__).resolve().parent
CORPUS = Path(ROOT) / "task1_test_files_2026"
LABELS = Path(ROOT) / "task1_test_labels_2026.json"
OUT = HERE / "expY_numbers.json"

WINDOW = 250          # characters either side of the suppression marker
EXPECT_QUERIES = 400
EXPECT_GOLD = 1750

SUPPRESSION = re.compile(r"FRAGMENT_SUPPRESSED")

# Narrow: the clearest single signal of negative treatment.
NARROW = re.compile(r"\bdistinguish\w*", re.I)
# Wide: adds the other standard formulations.
WIDE = re.compile(
    r"(\bdistinguish\w*|declined to follow|doubt (?:has been )?cast"
    r"|overrul\w*|not persuaded by)", re.I)


def main():
    if not CORPUS.exists() or not LABELS.exists():
        sys.exit(f"Set COLIEE_ROOT to a copy of the licensed collection; "
                 f"expected {CORPUS} and {LABELS}")

    gold = json.load(open(LABELS))
    n_gold = sum(len(v) if isinstance(v, list)
                 else len([x for x in str(v).split(",") if x.strip()])
                 for v in gold.values())
    if len(gold) != EXPECT_QUERIES or n_gold != EXPECT_GOLD:
        sys.exit(f"GATE FAILED: {len(gold)} queries and {n_gold} gold citations, "
                 f"expected {EXPECT_QUERIES} and {EXPECT_GOLD}")

    counts = {"narrow": [], "wide": []}
    markers_total = 0
    missing = 0

    for qid in sorted(gold):
        name = qid if qid.endswith(".txt") else qid + ".txt"
        path = CORPUS / name
        if not path.exists():
            missing += 1
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        spans = list(SUPPRESSION.finditer(text))
        markers_total += len(spans)
        for label, pattern in (("narrow", NARROW), ("wide", WIDE)):
            for m in spans:
                context = text[max(0, m.start() - WINDOW): m.end() + WINDOW]
                if pattern.search(context):
                    counts[label].append(qid)
                    break

    if missing:
        sys.exit(f"GATE FAILED: {missing} query files not found under {CORPUS}")

    out = {
        "question": "how often does a Task 1 gold citation sit beside language "
                    "of distinguishing or negative treatment",
        "method": {
            "window_chars": WINDOW,
            "narrow_pattern": NARROW.pattern,
            "wide_pattern": WIDE.pattern,
            "unit": "query case, counted once if any marker matches",
            "caveat": "a lower bound on prevalence; detects only treatment "
                      "signalled by an explicit verb near the marker",
        },
        "gates": {"queries": len(gold), "gold_citations": n_gold},
        "suppression_markers_total": markers_total,
        "narrow": {
            "queries": len(counts["narrow"]),
            "share": round(len(counts["narrow"]) / len(gold), 4),
        },
        "wide": {
            "queries": len(counts["wide"]),
            "share": round(len(counts["wide"]) / len(gold), 4),
        },
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"narrow: {out['narrow']['queries']}/{len(gold)} "
          f"({out['narrow']['share']:.1%})")
    print(f"wide:   {out['wide']['queries']}/{len(gold)} "
          f"({out['wide']['share']:.1%})")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
