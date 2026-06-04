import sys, json, re, unicodedata, glob
import xml.etree.ElementTree as ET
from collections import Counter

def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")

KANJI_DIG = {'〇':0,'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
KANJI_UNIT = {'十':10,'百':100,'千':1000}

def kanji_to_int(k: str):
    k = norm(k).strip()
    if not k:
        return None
    if k.isdigit():
        return int(k)
    total = 0
    num = 0
    for ch in k:
        if ch in KANJI_DIG:
            num = KANJI_DIG[ch]
        elif ch in KANJI_UNIT:
            unit = KANJI_UNIT[ch]
            if num == 0:
                num = 1
            total += num * unit
            num = 0
    return total + num

# Match both forms:
#  - 第十五条 / 第四百六十五条の八
#  - 第五百七十二条 / 第三百九十八条の二十
# and also the no-leading-第 form:
#  - 五百七十二条 / 三百九十八条の二十  (just in case)
ART_PAT = re.compile(r'(?:第)?([一二三四五六七八九十百千〇零]+)条(?:の([一二三四五六七八九十〇零]+))?')

def extract_relevant_articles_from_t1(t1_text: str):
    t = norm(t1_text)
    rel = set()
    for m in ART_PAT.finditer(t):
        base = kanji_to_int(m.group(1))
        if base is None:
            continue
        sub = m.group(2)
        if sub:
            sub_i = kanji_to_int(sub)
            if sub_i is not None:
                rel.add(f"{base}-{sub_i}")
            else:
                rel.add(str(base))
        else:
            rel.add(str(base))
    return rel

def load_gold(train_folder: str):
    files = sorted(glob.glob(train_folder.rstrip("/") + "/riteval_*_jp.xml"))
    if not files:
        raise SystemExit(f"No riteval xml files found under: {train_folder}")

    gold = {}
    for fp in files:
        root = ET.parse(fp).getroot()
        for pair in root.findall("pair"):
            qid = pair.get("id")
            lab = (pair.get("label") or "").strip().upper()
            if not qid or lab not in ("Y","N"):
                continue
            t1_el = pair.find("t1")
            t1_text = "".join(t1_el.itertext()) if t1_el is not None else ""
            rel = extract_relevant_articles_from_t1(t1_text)
            # store
            gold[qid] = {"label": lab, "relevant": rel}

    return gold, files

def load_run_jsonl(path: str):
    run = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            qid = r.get("id")
            if not qid:
                continue
            pred = (r.get("pred_label") or "").strip().upper()
            if pred not in ("Y","N"):
                pred = "Y" if int(r.get("pred_int", 0)) == 1 else "N"
            retrieved = r.get("retrieved_arts") or []
            retrieved = [norm(x).strip() for x in retrieved]
            run[qid] = {"pred": pred, "retrieved": retrieved}
    return run

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 eval_task3_official.py <train_folder> <run.jsonl> [K]")
        sys.exit(1)

    train_folder = sys.argv[1]
    run_path = sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    gold, files = load_gold(train_folder)
    run = load_run_jsonl(run_path)

    C = Counter()
    P_sum = 0.0
    R_sum = 0.0
    n = 0

    # evaluate only overlapping ids
    for qid, g in gold.items():
        if qid not in run:
            continue
        rel = g["relevant"]
        if not rel:
            continue

        retrieved = run[qid]["retrieved"]
        if K is not None:
            retrieved = retrieved[:K]

        pred = run[qid]["pred"]
        label_correct = 1 if pred == g["label"] else 0

        hit = len(set(retrieved) & rel)
        prec = hit / len(retrieved) if retrieved else 0.0
        rec  = hit / len(rel) if rel else 0.0

        recall1 = 1 if rel.issubset(set(retrieved)) else 0

        C["queries"] += 1
        C["label_correct"] += label_correct
        C["recall1"] += recall1
        C["gated_correct"] += (label_correct and recall1)

        P_sum += prec
        R_sum += rec
        n += 1

    print("=== Task3 Official-aligned eval (t1-derived relevant set) ===")
    print("train_files:", len(files))
    print("gold_queries:", len(gold))
    print("run_queries:", len(run))
    print("queries_eval:", C["queries"])

    if C["queries"] == 0:
        print("No overlap or all relevant sets empty.")
        sys.exit(0)

    label_acc = C["label_correct"] / C["queries"]
    recall1_rate = C["recall1"] / C["queries"]
    gated_acc = C["gated_correct"] / C["queries"]

    P = P_sum / n if n else 0.0
    R = R_sum / n if n else 0.0
    F2 = (5*P*R/(4*P+R)) if (4*P+R)>0 else 0.0

    print("label_acc:", f"{label_acc:.4f}")
    print("recall1_rate:", f"{recall1_rate:.4f}")
    print("gated_acc:", f"{gated_acc:.4f}")
    print("P_macro:", f"{P:.4f}")
    print("R_macro:", f"{R:.4f}")
    print("F2:", f"{F2:.4f}")

if __name__ == "__main__":
    main()
