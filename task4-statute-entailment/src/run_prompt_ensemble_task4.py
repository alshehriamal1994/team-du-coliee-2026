#!/usr/bin/env python3
"""
Prompt Ensemble for COLIEE Task 4.
Uses multiple diverse prompt templates with the same model and votes.
Each prompt focuses on a different reasoning strategy.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


# ─── Diverse prompt templates ───

PROMPTS = [
    # Prompt 1: Standard reasoning (our best prompt)
    {
        "system": """あなたは日本の民法に関する法律専門家です。
与えられた条文（t1）だけを根拠として、陳述文（t2）が論理的に導かれるかどうかを判断してください。

判断の手順：
1. 条文の要件・条件を正確に読み取る
2. 例外規定・ただし書きを確認する
3. 否定の整合性（条文が「できない」「しない」等を含む場合）を確認する
4. 陳述文に含まれる量化表現・限定語を確認する
5. 条文から直接明示されていない場合でも、条文の論理的帰結（暗示的推論）として導かれる場合はYと判定する
6. 上記を踏まえ、陳述文が条文から導かれるならY、導かれない・矛盾する・情報不足ならNと判定する

重要：陳述文は約50%がYで約50%がNです。どちらかに偏らず、条文との論理的整合性のみで判断してください。
最後の行に必ずYまたはNだけを出力してください。""",
        "user": "【条文】\n{article}\n\n【陳述文】\n{statement}\n\n推論を行い、最後の行にYまたはNだけを書いてください。",
    },
    # Prompt 2: Focus on finding contradictions
    {
        "system": """あなたは日本の民法の専門家です。条文と陳述文を比較して、矛盾があるかを判定してください。

以下に注意：
- 陳述文が条文と矛盾する場合→N
- 陳述文が条文から論理的に導ける場合→Y
- 条文にない情報を前提とする場合→N
- 約50%がYで50%がNです

最後の行にYまたはNのみを出力。""",
        "user": "条文：\n{article}\n\n陳述文：\n{statement}\n\nこの陳述文は条文と矛盾しますか？矛盾がなく条文から導かれるならY、矛盾があるか導かれないならNで答えてください。",
    },
    # Prompt 3: Paraphrase check
    {
        "system": """あなたは法律文書の論理分析の専門家です。
条文の内容が陳述文の内容を含意（entail）するかを判定してください。
含意とは、条文が真であれば陳述文も必ず真であることを意味します。

Y = 条文が陳述文を含意する
N = 条文が陳述文を含意しない

最後の行にYかNだけ出力してください。""",
        "user": "【前提（条文）】\n{article}\n\n【仮説（陳述文）】\n{statement}\n\n前提は仮説を含意しますか？YまたはN。",
    },
    # Prompt 4: Strict legal analysis
    {
        "system": """あなたは司法試験の採点官です。受験者の解答（陳述文）が民法の条文に照らして正しいかを厳密に判定してください。

判定基準：
- 条文の要件を正確に満たしているか
- 例外規定やただし書きを見落としていないか
- 条件や期間の制限を正しく理解しているか
- 「すべて」「のみ」「必ず」などの限定語が条文と一致するか

正しければY、誤りがあればN。最後の行に判定のみ記載。""",
        "user": "参照条文：\n{article}\n\n受験者の解答：\n{statement}\n\nこの解答は条文に照らして正しいですか？最後の行にYかNで判定してください。",
    },
    # Prompt 5: Simple direct question
    {
        "system": "あなたは日本の民法の専門家です。条文に基づいて質問に答えてください。YかNで回答し、最後の行にY/Nのみ書いてください。",
        "user": "以下の条文を読んでください：\n{article}\n\n質問：次の文は上記の条文から導かれますか？\n「{statement}」\n\n最後の行にYまたはNだけを書いてください。",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--civil-xml", type=Path,
                        default=Path("DATA/train2026(1)/2026/civil.xml"))
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    arts = load_civil_articles(args.civil_xml)
    rows = load_jsonl(args.input_jsonl)

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

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.lora_path} ...", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    print(f"Model loaded. Using {len(PROMPTS)} prompt templates.", flush=True)

    with args.output.open("a", encoding="utf-8") as out_f:
        for i, row in enumerate(pending, start=1):
            qid = row["id"]
            article_text = resolve_article_text(row.get("articles", []), arts)
            statement = row.get("statement", "")

            votes = []
            for p_idx, prompt_template in enumerate(PROMPTS):
                messages = [
                    {"role": "system", "content": prompt_template["system"]},
                    {"role": "user", "content": prompt_template["user"].format(
                        article=article_text, statement=statement
                    )},
                ]

                try:
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    formatted = prompt_template["system"] + "\n\n" + \
                        prompt_template["user"].format(article=article_text, statement=statement)

                inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                new_tokens = out[0][inputs["input_ids"].shape[1]:]
                raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
                label = parse_label(raw_output)
                votes.append(label)

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
