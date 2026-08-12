#!/usr/bin/env python3
"""Single-call TAT-LLM-7B-FFT evaluation over poc_sample_v3.json (1,000 questions).

SOTA-comparison baseline: table + question in, answer out, one model call per
question -- no Calculator/Verifier, no retry (contrast with run_poc.py's
Condition 1/2/3/3b pipelines). Run download_tatllm.py first to fetch the model.

Prompt/output format is copied verbatim from the model's own SFT training data
(data/sft/end_to_end/tatqa/tatqa_dataset_dev.json in
https://github.com/fengbinzhu/TAT-LLM) -- that is the format the model was
fine-tuned on, so it is what it expects and reliably reproduces.

Scoring reuses the official TAT-QA metric (TAT-QA/tatqa_metric.py), same
convention as run_poc.py's score_record: pred_scale is whatever the model itself
predicts (mapped 'none' -> '' to match the gold scale vocabulary).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "TAT-QA"))
from tatqa_metric import TaTQAEmAndF1  # noqa: E402

SAMPLE_PATH = ROOT / "poc_sample_v3.json"
RESULTS_DIR = ROOT / "results"
DEFAULT_MODEL_DIR = ROOT / "models" / "tat-llm-7b-fft"
MAX_PROMPT_TOKENS = 4096

# TAT-LLM's own output vocabulary for scale -> TAT-QA gold "scale" field vocabulary.
SCALE_MAP = {"none": "", "percent": "percent", "thousand": "thousand", "million": "million", "billion": "billion"}

INSTRUCTION_HEADER = """Below is an instruction that describes a question answering task in the finance domain, paired with an input table and its relevant text that provide further context. The given question is relevant to the table and text. Generate an appropriate answer to the given question.



### Instruction

Given a table and a list of texts in the following, what is the answer to the question?

Please predict the answer and store it in a variable named `{answer}`. If there are multiple values, separate them using the '#' symbol.

If the value of the `{answer}` is numerical, predict its scale and store it in a variable named `{scale}`.

The value of `{scale}` can be one of the following: `none`, `percent`, `thousand`, `million`, or `billion`. For non-numerical values, set the value of `{scale}` to 'none'.



Finally, present the final answer in the format of "The answer is: {answer} #### and its corresponding scale is: {scale}"



### Table

"""

ANSWER_RE = re.compile(
    r"The answer is:\s*(.*?)\s*####\s*and its corresponding scale is:\s*([a-zA-Z]+)", re.DOTALL
)


def format_table_row(cells) -> str:
    return "|" + "".join(f"{c} |" for c in cells)


def format_table(rows) -> str:
    return "\n".join(format_table_row(r) for r in rows)


def format_text(paragraphs: list) -> str:
    ordered = sorted(paragraphs, key=lambda p: p.get("order", 0))
    return "\n".join(f"{p.get('order', i + 1)} {p['text']}" for i, p in enumerate(ordered))


def build_prompt(doc: dict, question: str) -> str:
    table_text = format_table(doc["table"]["table"])
    text_block = format_text(doc.get("paragraphs", []))
    return (
        INSTRUCTION_HEADER + table_text
        + "\n\n\n\n### Text\n\n" + text_block
        + "\n\n\n\n### Question \n\n" + question
    )


def parse_prediction(generated: str):
    """Returns (predicted_answer, predicted_scale). predicted_answer is a string,
    or a list of strings if the model separated multiple values with '#' (its own
    multi-span convention). Returns (None, '') if the expected format wasn't found."""
    m = ANSWER_RE.search(generated)
    if not m:
        return None, ""
    answer_str = m.group(1).strip()
    scale = SCALE_MAP.get(m.group(2).strip().lower(), "")
    parts = [p.strip() for p in answer_str.split("#") if p.strip()]
    if not parts:
        return None, scale
    return (parts[0] if len(parts) == 1 else parts), scale


def score_record(meter: TaTQAEmAndF1, rec: dict, gold_q: dict):
    meter(ground_truth=gold_q, prediction=rec["predicted_answer"], pred_scale=rec["predicted_scale"])
    rec["is_correct"] = bool(meter.get_raw()[-1]["em"] == 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=str(SAMPLE_PATH), help="path to sample json file")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                         help="local path (see download_tatllm.py) or HF repo id of the model")
    parser.add_argument("--limit", type=int, default=None, help="limit number of questions (smoke test)")
    parser.add_argument("--start", type=int, default=0, help="slice start index (for chunked/resumed runs)")
    parser.add_argument("--end", type=int, default=None, help="slice end index (for chunked/resumed runs)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--tag", default="", help="suffix appended to output filenames")
    parser.add_argument("--dry-run", action="store_true",
                         help="print the first 3 prompts and exit, skip loading the model")
    args = parser.parse_args()
    tag_suffix = f"_{args.tag}" if args.tag else ""

    with Path(args.sample).open(encoding="utf-8") as f:
        docs = json.load(f)
    docs = docs[args.start:args.end]
    if args.limit:
        docs = docs[:args.limit]

    if args.dry_run:
        for doc in docs[:3]:
            q = doc["questions"][0]
            print(build_prompt(doc, q["question"]))
            print("=" * 80)
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading tokenizer/model from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    meter = TaTQAEmAndF1()
    out_path = RESULTS_DIR / f"tatllm_eval{tag_suffix}.jsonl"
    n_correct = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, doc in enumerate(docs, 1):
            q = doc["questions"][0]
            prompt = build_prompt(doc, q["question"])
            t0 = time.perf_counter()

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=MAX_PROMPT_TOKENS - args.max_new_tokens,
            ).to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            predicted_answer, predicted_scale = parse_prediction(generated)
            rec = {
                "uid": q["uid"],
                "table_uid": doc["table"]["uid"],
                "gold": q,
                "raw_output": generated,
                "predicted_answer": predicted_answer,
                "predicted_scale": predicted_scale,
                "is_correct": False,
                "latency_seconds": time.perf_counter() - t0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            score_record(meter, rec, q)
            n_correct += int(rec["is_correct"])
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

            if i % 10 == 0 or i == len(docs):
                print(f"[{i}/{len(docs)}] running EM={n_correct / i:.3f}")

    em, f1, scale_score, op_score = meter.get_overall_metric()
    summary = {
        "model_dir": args.model_dir,
        "n": len(docs),
        "em": em,
        "f1": f1,
        "scale_em": scale_score,
        "op_em": op_score,
    }
    summary_path = RESULTS_DIR / f"tatllm_summary{tag_suffix}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path} and {summary_path}")


if __name__ == "__main__":
    main()
