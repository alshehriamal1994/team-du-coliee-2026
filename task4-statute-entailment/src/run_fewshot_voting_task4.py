#!/usr/bin/env python3
"""
Few-shot prompting with similarity-based example selection + majority voting.
Based on KIS team approach (COLIEE 2025 winner, 90.41%).

For each test question:
1. Find most similar training examples using embedding similarity
2. Select balanced Y/N examples from top-k similar
3. Run inference K times with temperature sampling
4. Majority vote for final prediction

Usage:
  python3 scripts/run_fewshot_voting_task4.py \
    --model-path models/elyza/Llama-3-ELYZA-JP-8B \
    --train-jsonl experiments/datasets/H30_formal/train.jsonl \
    --input-jsonl experiments/datasets/H30_formal/test.jsonl \
    --civil-xml DATA/train2026(1)/2026/civil.xml \
    --run-tag FSVOTE-ELYZA-H30 \
    --output experiments/runs/task4-H30.FSVOTE-ELYZA-H30 \
    --few-shot-n 4 --num-votes 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer


def load_civil_articles(xml_path: Path) -> dict[str, str]:
    tree = ET.parse(xml_path)
    arts: dict[str, str] = {}
    for a in tree.getroot().findall(".//Article"):
        num = a.get("num")
        if not num:
            continue
        cap = (a.findtext("caption") or "").strip()
        txt = (a.findtext("text") or "").strip()
        full = (cap + "\n" + txt).strip() if cap else txt
        if full:
            arts[num] = full
    return arts


def resolve_article_text(article_nums: list[str], arts: dict[str, str]) -> str:
    blocks = []
    for num in article_nums:
        text = arts.get(num)
        if text:
            blocks.append(f"【第{num}条】\n{text}")
        else:
            blocks.append(f"【第{num}条】（条文テキスト未収録）")
    return "\n\n".join(blocks)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    pat = re.compile(r"^(\S+) [YN] \S+$")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                done.add(m.group(1))
    return done


# ─── Prompt ───

SYSTEM_PROMPT = """あなたは日本の民法に関する法律専門家です。
与えられた条文（t1）だけを根拠として、陳述文（t2）が論理的に導かれるかどうかを判断してください。

判断の手順：
1. 条文の要件・条件を正確に読み取る
2. 例外規定・ただし書きを確認する
3. 否定の整合性（条文が「できない」「しない」等を含む場合）を確認する
4. 陳述文に含まれる量化表現・限定語を確認する
5. 条文から直接明示されていない場合でも、条文の論理的帰結（暗示的推論）として導かれる場合はYと判定する
6. 上記を踏まえ、陳述文が条文から導かれるならY、導かれない・矛盾する・情報不足ならNと判定する

重要：陳述文は約50%がYで約50%がNです。どちらかに偏らず、条文との論理的整合性のみで判断してください。
最後の行に必ずYまたはNだけを出力してください。"""


def build_fewshot_prompt(article_text: str, statement: str,
                          examples: list[dict]) -> list[dict]:
    """Build chat messages with few-shot examples."""
    parts = []
    if examples:
        parts.append("以下はいくつかの例です：\n")
        for ex in examples:
            parts.append(
                f"【条文】\n{ex['article_text']}\n\n"
                f"【陳述文】\n{ex['statement']}\n\n"
                f"【判定】\n{ex['label']}\n"
                f"{'─' * 40}"
            )
        parts.append("\n次の問題を判定してください：\n")

    parts.append(
        f"【条文】\n{article_text}\n\n"
        f"【陳述文】\n{statement}\n\n"
        "推論を行い、最後の行にYまたはNだけを書いてください。"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]
    return messages


YN_LAST_LINE = re.compile(r"^([YN])\s*$", re.MULTILINE | re.IGNORECASE)
YN_ANYWHERE = re.compile(r"\b([YN])\b", re.IGNORECASE)


def parse_label(raw: str) -> str:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1].upper()
        if last in ("Y", "N"):
            return last
        if last.startswith("YES") or last == "はい":
            return "Y"
        if last.startswith("NO") or last == "いいえ":
            return "N"
    for line in reversed(lines):
        m = re.match(r"^[YN]$", line.strip().upper())
        if m:
            return line.strip().upper()
    tokens = YN_ANYWHERE.findall(raw)
    if tokens:
        return tokens[-1].upper()
    return "N"


# ─── Similarity-based example selection ───

class FewShotSelector:
    """Select balanced few-shot examples based on embedding similarity."""

    def __init__(self, train_rows: list[dict], arts: dict[str, str],
                 embed_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.train_rows = train_rows
        self.arts = arts

        print(f"Loading embedding model: {embed_model_name} ...", flush=True)
        self.embed_model = SentenceTransformer(embed_model_name)

        # Pre-compute embeddings for all training examples
        print(f"Computing embeddings for {len(train_rows)} training examples ...", flush=True)
        self.train_texts = []
        self.train_labels = []
        for row in train_rows:
            art_text = resolve_article_text(row.get("articles", []), arts)
            stmt = row.get("statement", "")
            # Use statement + article numbers as embedding text (concise)
            art_nums = ", ".join(row.get("articles", []))
            self.train_texts.append(f"条{art_nums}: {stmt}")
            self.train_labels.append(row.get("label", "N"))

        self.train_embeddings = self.embed_model.encode(
            self.train_texts, batch_size=64, show_progress_bar=True,
            normalize_embeddings=True
        )
        print("  Embeddings computed.", flush=True)

    def select(self, row: dict, n: int = 4) -> list[dict]:
        """Select n balanced (Y/N) few-shot examples most similar to the query."""
        art_nums = ", ".join(row.get("articles", []))
        query_text = f"条{art_nums}: {row.get('statement', '')}"
        query_emb = self.embed_model.encode([query_text], normalize_embeddings=True)

        # Compute similarities
        sims = (query_emb @ self.train_embeddings.T)[0]

        # Split by label and sort by similarity
        y_indices = [(i, sims[i]) for i in range(len(self.train_rows))
                     if self.train_labels[i] == "Y"]
        n_indices = [(i, sims[i]) for i in range(len(self.train_rows))
                     if self.train_labels[i] == "N"]

        y_indices.sort(key=lambda x: x[1], reverse=True)
        n_indices.sort(key=lambda x: x[1], reverse=True)

        # Select balanced examples
        half = n // 2
        selected_indices = [idx for idx, _ in y_indices[:half]] + \
                          [idx for idx, _ in n_indices[:half]]

        examples = []
        for idx in selected_indices:
            r = self.train_rows[idx]
            art_text = resolve_article_text(r.get("articles", []), self.arts)
            examples.append({
                "article_text": art_text,
                "statement": r.get("statement", ""),
                "label": r.get("label", "N"),
            })
        return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, default=None,
                        help="Optional LoRA adapter path")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--civil-xml", type=Path,
                        default=Path("DATA/train2026(1)/2026/civil.xml"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--few-shot-n", type=int, default=4)
    parser.add_argument("--num-votes", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--embed-model", type=str,
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    # Load articles
    arts = load_civil_articles(args.civil_xml)
    print(f"Loaded {len(arts)} articles.", flush=True)

    # Load training data and build selector
    train_rows = load_jsonl(args.train_jsonl)
    selector = FewShotSelector(train_rows, arts, args.embed_model)

    # Load test data
    rows = load_jsonl(args.input_jsonl)

    # Resume support
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(args.output)
    pending = [r for r in rows if r["id"] not in done_ids]
    if done_ids:
        print(f"Resuming: {len(done_ids)} done, {len(pending)} remaining.", flush=True)
    if not pending:
        print("All done.", flush=True)
        return

    # Load model
    print(f"Loading model from {args.model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    # Load LoRA adapter if specified
    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.lora_path} ...", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    print("Model loaded.", flush=True)

    # Inference with voting
    with args.output.open("a", encoding="utf-8") as out_f:
        for i, row in enumerate(pending, start=1):
            qid = row["id"]
            article_text = resolve_article_text(row.get("articles", []), arts)
            statement = row.get("statement", "")

            # Select similar few-shot examples
            examples = selector.select(row, n=args.few_shot_n)

            # Build prompt
            messages = build_fewshot_prompt(article_text, statement, examples)

            # Vote across multiple inference passes
            votes = []
            for v in range(args.num_votes):
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(formatted, return_tensors="pt")
                inputs = {k: v_.to(model.device) for k, v_ in inputs.items()}

                with torch.no_grad():
                    if args.num_votes == 1:
                        # Deterministic for single vote
                        out = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    else:
                        out = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                new_tokens = out[0][inputs["input_ids"].shape[1]:]
                raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
                label = parse_label(raw_output)
                votes.append(label)

            # Majority vote
            counter = Counter(votes)
            final_label = counter.most_common(1)[0][0]

            out_f.write(f"{qid} {final_label} {args.run_tag}\n")
            out_f.flush()

            if i % 5 == 0 or i == len(pending):
                vote_str = "".join(votes)
                print(
                    f"  [{i}/{len(pending)}] {qid} → {final_label} (votes: {vote_str})",
                    flush=True,
                )

    print(f"\nDone. Predictions written to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
