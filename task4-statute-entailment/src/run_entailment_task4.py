#!/usr/bin/env python3
"""
COLIEE Task 4, legal textual entailment inference.
Loads a local HuggingFace model, looks up article text from civil.xml, builds a
Japanese chain-of-thought prompt, and writes submission-format predictions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# article loader

def load_civil_articles(xml_path: Path) -> dict[str, str]:
    """Parse civil.xml → {article_num: full_text}."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    arts: dict[str, str] = {}
    for a in root.findall(".//Article"):
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
    """Return concatenated article text for the given article numbers."""
    blocks = []
    for num in article_nums:
        text = arts.get(num)
        if text:
            blocks.append(f"【第{num}条】\n{text}")
        else:
            blocks.append(f"【第{num}条】（条文テキスト未収録）")
    return "\n\n".join(blocks)

# prompt builder

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


def build_user_message(article_text: str, statement: str, few_shots: list[dict]) -> str:
    """Build the user-side message, optionally prepending few-shot examples."""
    parts: list[str] = []

    if few_shots:
        parts.append("以下はいくつかの例です：\n")
        for ex in few_shots:
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
    return "\n".join(parts)


def build_raw_prompt(article_text: str, statement: str, few_shots: list[dict]) -> str:
    """Build a raw (non-chat-template) prompt."""
    user_msg = build_user_message(article_text, statement, few_shots)
    return f"{SYSTEM_PROMPT}\n\n{user_msg}\n\n判定："

# output parser

YN_LAST_LINE = re.compile(r"^([YN])\s*$", re.MULTILINE | re.IGNORECASE)
YN_ANYWHERE = re.compile(r"\b([YN])\b", re.IGNORECASE)


def parse_label(raw: str) -> str:
    """Extract Y/N from model output. Prefer last-line, fallback to last occurrence."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    # check last line first
    if lines:
        last = lines[-1].upper()
        if last in ("Y", "N"):
            return last
        if last.startswith("YES") or last == "はい":
            return "Y"
        if last.startswith("NO") or last == "いいえ":
            return "N"
    # search from bottom up for standalone Y/N
    for line in reversed(lines):
        m = re.match(r"^[YN]$", line.strip().upper())
        if m:
            return line.strip().upper()
    # fallback: last occurrence of Y or N as token
    tokens = YN_ANYWHERE.findall(raw)
    if tokens:
        return tokens[-1].upper()
    return "N"  # safe default

# jsonl helpers

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_ids(path: Path) -> set[str]:
    """Read already-written prediction IDs for resume support."""
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


def select_few_shots(
    train_jsonl: Path,
    arts: dict[str, str],
    n: int = 4,
) -> list[dict]:
    """
    Select n balanced (Y/N) few-shot examples from training data.
    Picks the shortest examples for minimal token usage.
    """
    rows = load_jsonl(train_jsonl)
    y_rows = [r for r in rows if r.get("label") == "Y"]
    n_rows = [r for r in rows if r.get("label") == "N"]
    # sort by statement length (shorter = less tokens)
    y_rows.sort(key=lambda r: len(r.get("statement", "")))
    n_rows.sort(key=lambda r: len(r.get("statement", "")))
    half = n // 2
    selected = y_rows[:half] + n_rows[:half]
    shots = []
    for r in selected:
        art_text = resolve_article_text(r.get("articles", []), arts)
        shots.append({
            "article_text": art_text,
            "statement": r.get("statement", ""),
            "label": r.get("label", "N"),
        })
    return shots

# model loader

def load_model_and_tokenizer(
    model_path: Path,
    load_in_4bit: bool,
    load_in_8bit: bool,
):
    print(f"Loading tokenizer from {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model ({model_path.name}) ...", flush=True)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    print(f"Model loaded. device_map: {getattr(model, 'hf_device_map', 'auto')}", flush=True)
    return model, tokenizer

# inference

def run_inference(
    model,
    tokenizer,
    prompt_text: str,
    use_chat_template: bool,
    max_new_tokens: int,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # fallback if template doesn't support system role
            formatted = prompt_text
        inputs = tokenizer(formatted, return_tensors="pt")
    else:
        inputs = tokenizer(prompt_text, return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# main

def main() -> None:
    parser = argparse.ArgumentParser(description="COLIEE Task 4 entailment inference")
    parser.add_argument("--model-path", type=Path, required=True, help="Local HF model directory")
    parser.add_argument("--civil-xml", type=Path,
                        default=Path("../data/task4/civil_code.xml"),
                        help="Path to civil.xml article database")
    parser.add_argument("--input-jsonl", type=Path, required=True,
                        help="Input JSONL with fields: id, articles (list of nums), statement")
    parser.add_argument("--run-tag", required=True, help="Run tag string for submission format")
    parser.add_argument("--output", type=Path, required=True, help="Output prediction file")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max tokens to generate (use 256+ for CoT, 4 for direct Y/N)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (requires bitsandbytes)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit (requires bitsandbytes)")
    parser.add_argument("--use-chat-template", action="store_true",
                        help="Use tokenizer chat template for instruct models")
    parser.add_argument("--few-shot-jsonl", type=Path, default=None,
                        help="Training JSONL for few-shot examples (optional)")
    parser.add_argument("--few-shot-n", type=int, default=4,
                        help="Number of few-shot examples to prepend")
    parser.add_argument("--limit", type=int, default=0,
                        help="Debug: process only first N rows")
    args = parser.parse_args()

    # Load article database
    if not args.civil_xml.exists():
        sys.exit(f"ERROR: civil.xml not found at {args.civil_xml}")
    print(f"Loading civil.xml from {args.civil_xml} ...", flush=True)
    arts = load_civil_articles(args.civil_xml)
    print(f"  Loaded {len(arts)} articles.", flush=True)

    # Load few-shot examples
    few_shots: list[dict] = []
    if args.few_shot_jsonl:
        few_shots = select_few_shots(args.few_shot_jsonl, arts, n=args.few_shot_n)
        print(f"  Loaded {len(few_shots)} few-shot examples.", flush=True)

    # Load input data
    rows = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]

    # Resume support
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(args.output)
    pending = [r for r in rows if r["id"] not in done_ids]
    if done_ids:
        print(f"  Resuming: {len(done_ids)} already done, {len(pending)} remaining.", flush=True)

    if not pending:
        print("All done. Nothing to process.", flush=True)
        return

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # Inference loop
    with args.output.open("a", encoding="utf-8") as out_f:
        for i, row in enumerate(pending, start=1):
            qid = row["id"]
            article_nums = row.get("articles", [])
            statement = row.get("statement", "")

            article_text = resolve_article_text(article_nums, arts)
            user_msg = build_user_message(article_text, statement, few_shots)

            if args.use_chat_template:
                raw_output = run_inference(
                    model, tokenizer, user_msg,
                    use_chat_template=True,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                prompt = build_raw_prompt(article_text, statement, few_shots)
                raw_output = run_inference(
                    model, tokenizer, prompt,
                    use_chat_template=False,
                    max_new_tokens=args.max_new_tokens,
                )

            label = parse_label(raw_output)
            out_f.write(f"{qid} {label} {args.run_tag}\n")
            out_f.flush()

            if i % 10 == 0 or i == len(pending):
                print(
                    f"  [{i}/{len(pending)}] {qid} → {label}  |  "
                    f'raw: "{raw_output.strip()[:60]}"',
                    flush=True,
                )

    print(f"\nDone. Predictions written to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
