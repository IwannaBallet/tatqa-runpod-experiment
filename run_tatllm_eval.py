#!/usr/bin/env python3
"""Single-call TAT-LLM-7B-FFT evaluation over poc_sample_v3.json (1,000 questions).

SOTA-comparison baseline: table + question in, answer out, one model call per
question -- no Calculator/Verifier, no retry (contrast with run_poc.py's
Condition 1/2/3/3b pipelines). Run download_tatllm.py first to fetch the model.

Prompt is TAT-LLM's Step-wise Pipeline format, copied verbatim from the model's own
SFT training data (data/sft/stepwise/tatqa/tatqa_dataset_dev.json in
https://github.com/fengbinzhu/TAT-LLM) -- NOT the simpler data/sft/end_to_end
format. The model card (https://huggingface.co/next-tat/tat-llm-7b-fft) confirms
this is how it was trained: "crafted through the innovative Step-wise Pipeline
approach ... embodying three fundamental phases: Extraction, Reasoning, and
Execution." The model outputs a `| step | output |` table (question type /
evidence / equation / answer / scale) plus a final "The answer is: ..." sentence.

Output post-processing (the "External Executor" the model card refers to) is a
verbatim port of `parse_pred_answer` / `analyze_sft_response` / `clean_equation`
from tat_llm_eval.py in the repo above (dataset='tatqa' branch): for arithmetic
questions it re-runs the model's own equation (step 3) through eval() rather than
trusting the model's restated final number (step 4), and for count questions it
counts the evidence spans (step 2) rather than trusting the model's count. This
matters because LLMs occasionally state an equation correctly but misstate its
result. NOT ported: tat_llm_eval.py's `measure_match` scale-tolerant fuzzy number
matching (compares within 1% and across x100/x100 scale confusions before scoring)
-- that changes what counts as "correct" beyond just fixing arithmetic slips, and
run_poc.py's own convention (see its module docstring) is to score with the
official TAT-QA metric unmodified, so this script does the same for comparability
with our own Condition 1/2/3/3b results.

Scoring reuses the official TAT-QA metric (TAT-QA/tatqa_metric.py), same
convention as run_poc.py's score_record.
"""

from __future__ import annotations

import argparse
import json
import string
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

INSTRUCTION_HEADER = """Below is an instruction that describes a question answering task in the finance domain, paired with an input table and its relevant text that provide further context. The given question is relevant to the table and text. Generate an appropriate answer to the given question.



### Instruction

Given a table and a list of texts in the following, answer the question posed using the following five-step process:

1. Step 1: Predict the type of question being asked. Store this prediction in the variable `{question_type}`. The value of `{question_type}` can be one of the following:`Single span`, `Multiple spans`, `Count`, or `Arithmetic`.

2. Step 2: Extract the relevant strings or numerical values from the provided table or texts. Store these pieces of evidence in the variable `{evidence}`. If there are multiple pieces of evidence, separate them using the '#' symbol.

3. Step 3: if the `{question_type}` is `Arithmetic`, formulate an equation using values stored in `{evidence}`. Store this equation in the variable `{equation}`. For all other question types, set the value of {equation} to 'N.A.'.

4. Step 4: Predict or calculate the answer based on the question type, evidence and equation. Store it in the variable `{answer}`. If there are multiple values, separate them using the '#' symbol.

5. Step 5: If the value of the `{answer}` is numerical, predict its scale and store it in a variable named `{scale}`. The value of `{scale}` can be one of the following: `none`, `percent`, `thousand`, `million`, or `billion`. For non-numerical values, set the value of `{scale}` to 'none'.

Please organize the results in the following table:

| step | output |
| 1 | {question_type} |
| 2 | {evidence} |
| 3 | {equation} |
| 4 | {answer} |
| 5 | {scale} |

Finally, present the final answer in the format: "The answer is: {answer} #### and its corresponding scale is: {scale}"

### Table

"""


def format_table_row(cells) -> str:
    return "|" + "".join(f"{c} |" for c in cells)


def format_table(rows) -> str:
    return "\n".join(format_table_row(r) for r in rows)


def format_text(paragraphs: list) -> str:
    ordered = sorted(paragraphs, key=lambda p: p.get("order", 0))
    return "\n".join(f"{p.get('order', i + 1)} {p['text']}" for i, p in enumerate(ordered))


def build_query_block(table_rows, paragraphs, question: str) -> str:
    return (
        INSTRUCTION_HEADER + format_table(table_rows)
        + "\n\n### Text\n\n" + format_text(paragraphs)
        + "\n\n### Question \n\n" + question
        + "\n\n### Response\n\n"
    )


# ---------------------------------------------------------------------------
# Few-shot examples -- two arithmetic questions pulled verbatim from TAT-LLM's
# own stepwise SFT training data (data/sft/stepwise/tatqa/tatqa_dataset_train.json
# in https://github.com/fengbinzhu/TAT-LLM), included so the model more reliably
# produces the 5-step table (question type -> evidence -> equation -> answer ->
# scale) that the Executor post-processing below depends on. Deliberately drawn
# from the TRAIN split and verified to have zero table/question overlap with
# poc_sample_v3.json (checked exact question-text and table-content matches) --
# using a dev-split example risked leaking one of our own 1,000 eval questions
# into the prompt (an earlier goodwill-impairment dev example we considered
# turned out to already be one of our eval questions).
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "table": [
            ["($ in millions except per share amounts)", "", "", ""],
            ["For the year ended December 31:*", "2019", "2018", "Yr.-to-Yr. Percent Change"],
            ["Net income as reported", "$ 9,431", "8,728*", "8.1%"],
            ["Income/(loss) from discontinued operations, net of tax", "(4)", "5", "NM"],
            ["Income from continuing operations", "$ 9,435", "8,723*", "8.2%"],
            ["Non-operating adjustments (net of tax)", "", "", ""],
            ["Acquisition-related charges", "1,343", "649", "107.0"],
            ["Non-operating retirement-related costs/(income)", "512", "1,248", "(58.9)"],
            ["U.S. tax reform charge", "146", "2,037", "(92.8)"],
            ["Operating (non-GAAP) earnings", "$11,436", "$12,657", "(9.6)%"],
            ["Diluted operating (non-GAAP) earnings per share", "$ 12.81", "$ 13.81", "(7.2)%"],
        ],
        "paragraphs": [
            {"order": 1, "text": "The following table provides the company's operating (non-GAAP) earnings "
                                  "for 2019 and 2018. See page 46 for additional information."},
            {"order": 2, "text": "* 2019 results were impacted by Red Hat purchase accounting and "
                                  "acquisition-related activity."},
            {"order": 3, "text": "** Includes charges of $2.0 billion in 2018 associated with U.S. tax reform."},
        ],
        "question": "What was the increase / (decrease) in Net income from 2018 to 2019?",
        "response": """| step | output |
| 1 | Arithmetic |
| 2 | 8728#9431 |
| 3 | 9431 - 8728 |
| 4 | 703 |
| 5 | million |

The answer is: 703 #### and its corresponding scale is: million""",
    },
    {
        "table": [
            ["", "Year ended December 31, 2019", "Year ended December 31, 2018", "Year ended December 31, 2017"],
            ["Income", "55", "47", "30"],
            ["Expense", "(54)", "(54)", "(52)"],
            ["Total", "1", "(7)", "(22)"],
        ],
        "paragraphs": [
            {"order": 1, "text": "Interest income (expense), net consisted of the following:"},
            {"order": 2, "text": "Interest income is related to the cash and cash equivalents held by the "
                                  "Company. Interest expense recorded in 2019, 2018 and 2017 included "
                                  "respectively a charge of $39 million, $38 million and $37 million on the "
                                  "senior unsecured convertible bonds issued in July 2017 and July 2014, of "
                                  "which respectively $37 million, $36 million and $33 million was a non-cash "
                                  "interest expense resulting from the accretion of the discount on the "
                                  "liability component. Net interest includes also charges related to the "
                                  "banking fees and the sale of trade and other receivables."},
            {"order": 3, "text": "No borrowing cost was capitalized in 2019, 2018 and 2017. Interest income on "
                                  "government bonds and floating rate notes classified as available-for-sale "
                                  "marketable securities amounted to $6 million for the year ended December 31, "
                                  "2019, $6 million for the year ended December 31, 2018 and $6 million for the "
                                  "year ended December 31, 2017."},
        ],
        "question": "What is the average Income?",
        "response": """| step | output |
| 1 | Arithmetic |
| 2 | 30#47#55 |
| 3 | ((55 + 47) + 30) / 3 |
| 4 | 44 |
| 5 | million |

The answer is: 44 #### and its corresponding scale is: million""",
    },
]

FEW_SHOT_BLOCK = "\n\n\n".join(
    build_query_block(ex["table"], ex["paragraphs"], ex["question"]) + ex["response"]
    for ex in FEW_SHOT_EXAMPLES
)


def build_prompt(doc: dict, question: str) -> str:
    query_block = build_query_block(doc["table"]["table"], doc.get("paragraphs", []), question)
    return FEW_SHOT_BLOCK + "\n\n\n" + query_block


# ---------------------------------------------------------------------------
# External Executor -- ported from tat_llm_eval.py (fengbinzhu/TAT-LLM)
# ---------------------------------------------------------------------------
def analyze_sft_response(llm_response_text: str) -> dict:
    """Parses the model's '| step | output |' table into {step_number: output_text}.
    Verbatim port of tat_llm_eval.py:analyze_sft_response -- any '|'-containing line
    in the (lower-cased) response becomes a row, keyed by its first column."""
    res_map = {}
    for line in llm_response_text.split("\n"):
        if "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|") if c.strip() != ""]
        if len(cols) < 2:
            continue
        res_map[cols[0]] = cols[1]
    return res_map


def clean_equation(equation: str) -> str:
    """Strips lowercase letters and currency/percent/comma noise so eval() sees a
    bare arithmetic expression. Verbatim port of tat_llm_eval.py:clean_equation."""
    strip_chars = set(string.ascii_lowercase) | {"&", "%", ",", "$"}
    return "".join(c for c in equation if c not in strip_chars)


def parse_prediction(generated: str, gold_q: dict, log_uid: str = ""):
    """Returns (predicted_answer, predicted_scale), ported from tat_llm_eval.py's
    parse_pred_answer(dataset='tatqa'). predicted_answer is a string, or a list of
    strings for multi-span gold questions."""
    text = generated.lower()
    pred_scale = ""
    llm_ans_str = text.strip()

    if "the answer is:" in text:
        llm_ans_str = text.split("the answer is:")[1].strip().replace("</s>", "")
        arr = llm_ans_str.split("####")
        llm_ans_str = arr[0].strip()
        if len(arr) > 1:
            pred_scale = arr[1].replace("and its corresponding scale is:", "").strip()
            pred_scale = "" if pred_scale == "none" else pred_scale

    res_map = analyze_sft_response(text)
    question_type = res_map.get("1")
    try:
        if question_type == "arithmetic" and "3" in res_map:
            llm_ans_str = str(round(eval(clean_equation(res_map["3"])), 4))  # noqa: S307
        elif question_type == "count" and "2" in res_map:
            llm_ans_str = str(len(res_map["2"].strip().split("#")))
        elif question_type == "multiple spans" and "2" in res_map:
            llm_ans_str = res_map["2"].strip()
        elif question_type == "single span" and "2" in res_map:
            llm_ans_str = res_map["2"].strip()
    except Exception as e:
        print(f"[executor] equation eval failed for uid={log_uid}: {e}", file=sys.stderr)

    if not llm_ans_str:
        return None, pred_scale

    if gold_q.get("answer_type") == "multi-span":
        parts = [p.strip() for p in llm_ans_str.split("#") if p.strip()]
        return (parts or None), pred_scale
    return llm_ans_str, pred_scale


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
    parser.add_argument("--max-new-tokens", type=int, default=256,
                         help="stepwise output (5-row table + summary sentence) needs more headroom than a bare answer")
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
    # Truncate from the left (drop few-shot content first) so an oversized real
    # table never costs us the actual question / "### Response" cue at the end.
    tokenizer.truncation_side = "left"
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

            predicted_answer, predicted_scale = parse_prediction(generated, q, log_uid=q["uid"])
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
