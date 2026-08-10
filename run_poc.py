#!/usr/bin/env python3
"""PoC runner for Calculator/Verifier prompt spec (PROMPT_SPEC.md, PoC section only).

Runs Condition 1 (Standard), Condition 2 (PoT), Condition 3 (Proposed: Calculator +
Verifier with one retry) over poc_sample.json (80 questions, seed=7 from
filtered_dev.json) for each of the two local Ollama models. Writes per-attempt
JSONL logs to results/, plus results/poc_summary.json and results/audit_sample.jsonl
(verifier-oracle audit sample, see verifier_oracle_spec.md).

EM scoring uses the official TAT-QA metric (TAT-QA/tatqa_metric.py) unmodified.
Since none of the three conditions asks the model to predict a "scale" (only
"answer"), pred_scale is set to the gold question's own scale field -- the
model's numeric answer is produced directly from raw table cell values, the
same convention the gold "answer" field uses, so both sides must be scaled by
the same factor to be compared correctly. This does not "cheat" EM (which
remains a real magnitude/exact-match check); it just means this PoC does not
separately score scale-detection, matching the fact that no condition's
output schema includes a "scale" field.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import builtins
import json
import random
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "TAT-QA"))
from tatqa_metric import TaTQAEmAndF1  # noqa: E402

SAMPLE_PATH = ROOT / "filtered_tatqa" / "poc_sample.json"
RESULTS_DIR = ROOT / "results"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODELS = {
    "qwen2.5:3b-instruct": "qwen",
    "llama3.2:3b": "llama",
    "qwen3:1.7b": "qwen3-1.7b",
    "qwen3:4b": "qwen3-4b",
    "qwen3:4b-instruct": "qwen3-4b",
    "qwen3:8b": "qwen3-8b",
    "qwen3:32b": "qwen3-32b",
}

CODE_TIMEOUT_SEC = 5
UNICODE_MINUS = "−"

# ---------------------------------------------------------------------------
# Prompts (verbatim from PROMPT_SPEC.md)
# ---------------------------------------------------------------------------
CALCULATOR_SYSTEM_BASE = """당신은 재무제표 표를 보고 산술 질문에 답하는 어시스턴트입니다.
아래 절차를 따르세요:
1. 질문에 답하기 위해 표에서 필요한 항목과 값을 찾습니다.
2. 각 항목의 이름과 값을 명시적으로 나열합니다 (표에 적힌 그대로, 단위/부호 포함).
3. 이 값들을 사용한 계산을 파이썬 코드로 작성합니다.
4. 반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 추가하지 마세요.

{
  "used_values": {
    "<항목명1>": <값1>,
    "<항목명2>": <값2>
  },
  "code": "<파이썬 코드, 마지막 줄은 result = ... 형태>",
  "answer": <계산 결과>
}"""

CALCULATOR_USER_TEMPLATE = """표:
{table_as_text}

질문: {question}"""

CALCULATOR_RETRY_FEEDBACK_TEMPLATE = """아래 항목에서 값이 표와 일치하지 않는 것으로 확인되었습니다. 다시 확인하고 답하세요.
{mismatch_lines}
(위와 동일한 JSON 형식으로 재답변)"""

VERIFIER_SYSTEM_BASE = """당신은 다른 어시스턴트(Calculator)가 재무제표 표에서 값을 추출해 계산한 결과를 검증하는 검증자입니다.
Calculator가 어떤 계산을 했는지, 어떤 코드를 썼는지는 전달받지 않습니다.
오직 아래 정보만 받습니다: 원본 표, Calculator가 "사용했다"고 주장하는 항목명과 값의 목록.

절차:
1. 표를 처음부터 독립적으로 읽고, 주어진 각 항목명에 해당하는 실제 값을 스스로 찾습니다.
2. Calculator가 주장한 값과, 당신이 독립적으로 찾은 값을 항목별로 대조합니다.
3. 아래 JSON 형식으로만 답하세요.

{
  "verified_values": {
    "<항목명1>": <Verifier가 독립적으로 찾은 값1>,
    "<항목명2>": <Verifier가 독립적으로 찾은 값2>
  },
  "mismatches": ["<불일치한 항목명만 나열, 없으면 빈 배열>"]
}"""

VERIFIER_USER_TEMPLATE = """표:
{table_as_text}

다음은 다른 어시스턴트가 계산에 사용했다고 주장하는 값들입니다. 이 값들이 표와 일치하는지 독립적으로 확인하세요.
{calculator_used_values_json}"""

STANDARD_SYSTEM = """당신은 재무제표 표를 보고 질문에 답하는 어시스턴트입니다.
표를 보고 질문에 대한 최종 숫자 답만 제시하세요. 풀이 과정은 출력하지 마세요.
반드시 아래 JSON 형식으로만 답하세요.

{"answer": <숫자>}"""

STANDARD_USER_TEMPLATE = """표: {table_as_text}
질문: {question}"""


# ---------------------------------------------------------------------------
# Table / number utilities
# ---------------------------------------------------------------------------
def format_table(table_obj: dict) -> str:
    rows = table_obj.get("table", []) if isinstance(table_obj, dict) else table_obj
    lines = []
    for row in rows:
        cells = [str(c).replace("\n", " ") for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Few-shot examples (filtered_tatqa/few_shot_examples.json -- never part of the
# 80-question eval sample, see PROMPT_SPEC.md). Hand-worked used_values/code/
# verified_values for two of those documents, appended to the Calculator and
# Verifier system prompts so Condition 2/3 stop being pure zero-shot.
# ---------------------------------------------------------------------------
with (ROOT / "filtered_tatqa" / "few_shot_examples.json").open(encoding="utf-8") as _fs_f:
    _FEW_SHOT_DOCS = json.load(_fs_f)

_FS1_TABLE = format_table(_FEW_SHOT_DOCS[1]["table"])  # Leasehold improvements: simple diff
_FS1_Q = _FEW_SHOT_DOCS[1]["questions"][0]["question"]
_FS2_TABLE = format_table(_FEW_SHOT_DOCS[2]["table"])  # Other guarantees: "2019 average" idiom
_FS2_Q = _FEW_SHOT_DOCS[2]["questions"][0]["question"]

CALCULATOR_FEW_SHOT = f"""

아래는 참고 예시입니다 (실제 문제와는 무관한 별도 예시입니다):

예시 1:
표:
{_FS1_TABLE}

질문: {_FS1_Q}

답:
{{
  "used_values": {{
    "Leasehold improvements (January 3, 2020)": 203,
    "Leasehold improvements (December 28, 2018)": 206
  }},
  "code": "result = 203 - 206",
  "answer": -3
}}

예시 2 ("N년 평균"은 N년 값과 그 직전 연도 값의 평균을 뜻함에 유의):
표:
{_FS2_TABLE}

질문: {_FS2_Q}

답:
{{
  "used_values": {{
    "Other guarantees and contingent liabilities (2019)": 2943,
    "Other guarantees and contingent liabilities (2018)": 4036
  }},
  "code": "result = (2943 + 4036) / 2",
  "answer": 3489.5
}}"""

CALCULATOR_SYSTEM = CALCULATOR_SYSTEM_BASE + CALCULATOR_FEW_SHOT

VERIFIER_FEW_SHOT = f"""

아래는 참고 예시입니다 (실제 문제와는 무관한 별도 예시입니다):

표:
{_FS1_TABLE}

다음은 다른 어시스턴트가 계산에 사용했다고 주장하는 값들입니다. 이 값들이 표와 일치하는지 독립적으로 확인하세요.
{{"Leasehold improvements (January 3, 2020)": 203, "Leasehold improvements (December 28, 2018)": 206}}

답:
{{
  "verified_values": {{
    "Leasehold improvements (January 3, 2020)": 203,
    "Leasehold improvements (December 28, 2018)": 206
  }},
  "mismatches": []
}}"""

VERIFIER_SYSTEM = VERIFIER_SYSTEM_BASE + VERIFIER_FEW_SHOT


# ---------------------------------------------------------------------------
# Condition 3b: "literate" Verifier (verifier_redesign_literate_pilot.md).
# Unlike Condition 3a/VERIFIER_SYSTEM above (blind: only sees used_values),
# this Verifier also sees Calculator's code and the question, so it can catch
# a wrong calculation LOGIC even when every extracted value is individually
# correct (e.g. question asks for a "total" but code only sums a subset).
# ---------------------------------------------------------------------------
VERIFIER_SYSTEM_BASE_LITERATE = """당신은 다른 어시스턴트(Calculator)가 재무제표 표에서 값을 추출하고 계산한 결과를 검증하는 검증자입니다. 아래 정보를 모두 전달받습니다: 원본 표, 질문, Calculator가 "사용했다"고 주장하는 항목명과 값의 목록, 그리고 Calculator가 실제로 실행한 파이썬 코드.

절차:
1. 표를 처음부터 독립적으로 읽고, 주어진 각 항목명에 해당하는 실제 값을 스스로 찾습니다. Calculator가 주장한 값과 대조하여 값 자체의 불일치가 있는지 확인합니다.
2. Calculator의 code를 검토하여, 질문이 요구하는 계산과 code가 실제로 수행하는 연산이 일치하는지 확인합니다. 예를 들어 질문이 "총계"를 묻는데 code가 세부 항목 하나만 참조하거나, 질문이 "감소분"을 묻는데 code가 반대 부호로 계산하는 경우 등을 확인합니다.
3. 값 불일치와 로직 불일치를 구분하여 아래 JSON 형식으로만 답하세요.

{
  "verified_values": {
    "<항목명1>": <Verifier가 독립적으로 찾은 값1>,
    "<항목명2>": <Verifier가 독립적으로 찾은 값2>
  },
  "value_mismatches": ["<값 자체가 표와 다른 항목명만 나열, 없으면 빈 배열>"],
  "logic_mismatch": true,
  "logic_mismatch_reason": "<logic_mismatch가 true인 경우, 질문 의도와 code가 실제로 다른 이유를 한 문장으로. false면 null>"
}"""

VERIFIER_USER_TEMPLATE_LITERATE = """표:
{table_as_text}

질문: {question}

다음은 다른 어시스턴트가 계산에 사용했다고 주장하는 값과 실제로 실행한 코드입니다. 값이 표와 일치하는지, 그리고 코드가 질문이 요구하는 계산과 일치하는지 독립적으로 확인하세요.
사용한 값: {calculator_used_values_json}
실행한 코드:
{calculator_code}"""

VERIFIER_FEW_SHOT_LITERATE = f"""

아래는 참고 예시입니다 (실제 문제와는 무관한 별도 예시입니다):

표:
{_FS1_TABLE}

질문: {_FS1_Q}

다음은 다른 어시스턴트가 계산에 사용했다고 주장하는 값과 실제로 실행한 코드입니다. 값이 표와 일치하는지, 그리고 코드가 질문이 요구하는 계산과 일치하는지 독립적으로 확인하세요.
사용한 값: {{"Leasehold improvements (January 3, 2020)": 203, "Leasehold improvements (December 28, 2018)": 206}}
실행한 코드:
result = 203 - 206

답:
{{
  "verified_values": {{
    "Leasehold improvements (January 3, 2020)": 203,
    "Leasehold improvements (December 28, 2018)": 206
  }},
  "value_mismatches": [],
  "logic_mismatch": false,
  "logic_mismatch_reason": null
}}"""

VERIFIER_SYSTEM_LITERATE = VERIFIER_SYSTEM_BASE_LITERATE + VERIFIER_FEW_SHOT_LITERATE

CALCULATOR_RETRY_FEEDBACK_TEMPLATE_LITERATE = """아래 문제가 발견되었습니다.
{issue_lines}
이를 반영하여 코드를 다시 작성하고 답하세요.
(위와 동일한 JSON 형식으로 재답변)"""


def normalize_label(s) -> str:
    s = str(s).lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", s)


def oracle_parse_number(value):
    """Robust numeric parse for the verifier oracle only (NOT used for official
    EM scoring). tatqa_utils.negative_num_handle's paren regex ([\\d.\\s]+) breaks
    on comma-formatted negatives like "(3,709)", so we implement our own
    paren/comma/unicode-minus handling here per RESEARCH_CONTEXT.md's warning."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(UNICODE_MINUS, "-")
    # A lone dash/em-dash/en-dash is financial-statement convention for "zero" /
    # "no value", not "no data" -- treat it as 0.0 rather than un-parseable
    # (fix_oracle_judge_rescore.md: this was previously returned as None, which
    # made oracle_judge silently fail to compare it against any claimed value).
    if s in ("-", "–", "—", "‐", "‑"):
        return 0.0
    # Allow a currency symbol before the opening paren (e.g. "$(1,221)"), not just
    # a bare "(1,221)" -- the original anchor-only-at-"(" regex missed these,
    # silently reading 25/80 sample cells' negative parenthesized amounts as positive.
    is_paren_negative = bool(re.match(r"^[\$€£¥]?\s*\(.*\)$", s.strip()))
    cleaned = re.sub(r"[\$€£¥,%()]", "", s).strip()
    if cleaned in ("", "-", "+"):
        return None
    try:
        num = float(cleaned)
    except ValueError:
        m = re.search(r"[+-]?\d+(\.\d+)?", cleaned)
        if not m:
            return None
        num = float(m.group(0))
    if is_paren_negative and num > 0:
        num = -num
    return num


# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------
def repair_truncated_json(text: str) -> str:
    """Best-effort repair for JSON cut off mid-object by an early EOS token
    (observed with qwen2.5:3b-instruct: valid JSON up to the last value, then
    the model stops before emitting closing braces). Closes any dangling
    string literal, then appends the missing closing ]/} in the right order.
    Does NOT attempt to fix genuinely malformed content (e.g. bare Python
    identifiers, `None` instead of `null`) -- those should still fail to parse."""
    depth_curly = 0
    depth_square = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth_curly += 1
            elif ch == "}":
                depth_curly -= 1
            elif ch == "[":
                depth_square += 1
            elif ch == "]":
                depth_square -= 1
    repaired = text
    if in_string:
        repaired += '"'
    repaired += "]" * max(depth_square, 0)
    repaired += "}" * max(depth_curly, 0)
    return repaired


def extract_json(raw: str):
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text), None
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        return None, "no_json_object_found"
    end = text.rfind("}")
    if end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate), None
        except Exception:
            pass
    # Fallback: response was likely cut off by an early stop token before the
    # closing brace(s) were emitted. Try to repair and reparse.
    candidate = text[start:]
    try:
        return json.loads(repair_truncated_json(candidate)), None
    except Exception as e:
        return None, f"json_parse_error: {e}"


# ---------------------------------------------------------------------------
# Sandboxed code execution
# ---------------------------------------------------------------------------
SAFE_BUILTINS = {name: getattr(builtins, name) for name in [
    "abs", "round", "min", "max", "sum", "len", "int", "float", "pow",
    "sorted", "list", "tuple", "dict", "set", "str", "range", "enumerate",
    "zip", "bool", "True", "False", "None",
] if hasattr(builtins, name)}


def normalize_indentation(code: str) -> str:
    """Best-effort fallback for stray/inconsistent leading whitespace in LLM-generated
    code (observed: a single spurious leading space on an otherwise flat line, e.g.
    'x = 1\\n y = 2', which is a plain IndentationError since Python only allows an
    indent increase right after a block-opening ':' line). Rebuilds indentation from
    scratch in 4-space steps, only indenting after a line that opens a block."""
    lines = code.split("\n")
    out = []
    indent_stack = [0]
    prev_opens_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if prev_opens_block:
            indent_stack.append(indent_stack[-1] + 4)
        out.append(" " * indent_stack[-1] + stripped)
        prev_opens_block = stripped.endswith(":")
    return "\n".join(out)


def run_code(code: str, used_values: dict):
    code = (code or "").strip()
    code = re.sub(r"^```(?:python)?", "", code, flags=re.I).strip()
    code = re.sub(r"```$", "", code).strip()
    if not code:
        return None, "empty code"

    def _exec(src):
        ns = {"__builtins__": SAFE_BUILTINS, "used_values": used_values}
        exec(src, ns)
        if "result" not in ns:
            raise RuntimeError("code did not define a 'result' variable")
        return ns["result"]

    def _run(src):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_exec, src)
            try:
                return fut.result(timeout=CODE_TIMEOUT_SEC), None
            except concurrent.futures.TimeoutError:
                return None, f"timeout after {CODE_TIMEOUT_SEC}s"
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"

    dedented = textwrap.dedent(code)
    result, err = _run(dedented)
    if err is not None and err.startswith("IndentationError"):
        result, err = _run(normalize_indentation(dedented))
    return result, err


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def call_llm(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


CODE_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Numbers that legitimately show up in code for scale/sign/averaging arithmetic
# (e.g. "* 100" for a percent, "/ 2" for a two-period average, "* 1000" for a
# thousand-scale conversion) rather than being an invented, unsourced fact.
COMMON_CODE_CONSTANTS = {0.0, 1.0, 2.0, -1.0, 3.0, 4.0, 10.0, 12.0, 100.0,
                          1000.0, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9}


def check_internal_consistency(used_values: dict, code: str, tol: float = 1e-6):
    """Returns (is_consistent, unused_keys).

    Distinguishes two previously-conflated situations (poc_fixes: prep_work_before_sweep.md
    fix 1): a used_values key the code never touches is harmless bookkeeping (e.g. a
    value the Calculator noted for context but didn't end up needing) and must NOT fail
    the attempt -- it's reported via unused_keys for logging only. Only a literal number
    in code that matches neither any claimed used_values value nor a common scale/sign
    constant is a real inconsistency: that means code computed from an unsourced, invented
    fact (e.g. used_values={"Additions_2019": 39116, ...} but code="result = (73 - 41) / 1e6").
    """
    code = code or ""
    code_numbers = [float(m) for m in CODE_NUMBER_RE.findall(code)]

    claimed_numbers = []
    unused_keys = []
    for key, val in (used_values or {}).items():
        referenced = f'used_values["{key}"]' in code or f"used_values['{key}']" in code
        num = oracle_parse_number(val)
        if num is not None:
            claimed_numbers.append(num)
        literal_match = num is not None and any(abs(num - cn) < tol for cn in code_numbers)
        if not referenced and not literal_match:
            unused_keys.append(key)

    is_consistent = True
    for cn in code_numbers:
        if any(abs(cn - v) < tol for v in claimed_numbers):
            continue
        if any(abs(cn - c) < tol for c in COMMON_CODE_CONSTANTS):
            continue
        is_consistent = False
        break

    return is_consistent, unused_keys


def run_calculator(client, model, table_text, question, feedback=None):
    user = CALCULATOR_USER_TEMPLATE.format(table_as_text=table_text, question=question)
    if feedback:
        user = user + "\n\n" + feedback
    raw = call_llm(client, model, CALCULATOR_SYSTEM, user)
    parsed, err = extract_json(raw)
    call = {
        "used_values": {},
        "code": "",
        "raw_response": raw,
        "code_exec_result": None,
        "code_exec_error": None,
        "parse_error": err,
        "internal_inconsistency": False,
        "unused_keys": [],
    }
    if parsed is None or not isinstance(parsed, dict):
        call["code_exec_error"] = "no valid JSON response: " + (err or "unknown")
        return call
    uv = parsed.get("used_values")
    call["used_values"] = uv if isinstance(uv, dict) else {}
    call["code"] = parsed.get("code") or ""
    if not call["code"]:
        call["code_exec_error"] = "no 'code' field in response"
        return call
    result, exec_err = run_code(call["code"], call["used_values"])
    call["code_exec_result"] = result
    call["code_exec_error"] = exec_err
    if exec_err is None:
        is_consistent, unused_keys = check_internal_consistency(call["used_values"], call["code"])
        call["internal_inconsistency"] = not is_consistent
        call["unused_keys"] = unused_keys
    return call


def find_best_match_key(item_name, used_values_keys):
    """Fuzzy-match a Verifier-claimed mismatch item name against the Calculator's
    actual used_values keys (normalized exact match, else substring containment)."""
    norm_item = normalize_label(item_name)
    if not norm_item:
        return None
    for key in used_values_keys:
        if norm_item == normalize_label(key):
            return key
    for key in used_values_keys:
        norm_key = normalize_label(key)
        if norm_key and (norm_item in norm_key or norm_key in norm_item):
            return key
    return None


def values_equivalent(a, b) -> bool:
    """Whether two claimed/verified values are the same number once parsed with
    the same dash-as-zero convention used by oracle_judge (fix_oracle_judge_rescore.md) --
    e.g. Calculator's 0 vs Verifier's "—" for the same cell. Both must parse to a
    number; unparseable values are never considered equivalent."""
    na, nb = oracle_parse_number(a), oracle_parse_number(b)
    if na is None or nb is None:
        return False
    return abs(na - nb) < 1e-9


def validate_mismatches(used_values: dict, mismatches: list, verified_values: dict | None = None):
    """Split a Verifier's mismatches list into: valid (corresponds to a real
    used_values key and, if a verified_values entry exists, actually differs from
    what was used), invalid (item name not present in the table / never claimed
    by the Calculator, per poc_fixes_v3.md fix 2), and self_consistent (name
    matches, but the Verifier's own verified_value is numerically the same as the
    used_value once dash-as-zero normalized -- a self-contradictory flag that
    should not block acceptance or trigger a retry; see the f1457678 case from
    the 1,000-question sweep where "0" vs "—" for the same cell wrongly discarded
    an already-correct answer)."""
    keys = list(used_values.keys())
    valid, invalid, self_consistent = [], [], []
    for item in mismatches:
        key = find_best_match_key(item, keys)
        if key is None:
            invalid.append(item)
            continue
        if verified_values and item in verified_values and \
                values_equivalent(used_values.get(key), verified_values.get(item)):
            self_consistent.append(item)
            continue
        valid.append(item)
    return valid, invalid, self_consistent


def run_verifier(client, model, table_text, used_values: dict):
    user = VERIFIER_USER_TEMPLATE.format(
        table_as_text=table_text,
        calculator_used_values_json=json.dumps(used_values, ensure_ascii=False),
    )
    raw = call_llm(client, model, VERIFIER_SYSTEM, user)
    parsed, err = extract_json(raw)
    call = {
        "verified_values": {}, "mismatches": [], "valid_mismatches": [],
        "invalid_mismatch_items": [], "self_consistent_mismatch_items": [],
        "raw_response": raw, "parse_error": err,
    }
    if parsed is None or not isinstance(parsed, dict):
        # Fail-safe: can't determine mismatches, so treat everything as unverified
        # (forces the retry / Fail path rather than silently accepting).
        call["mismatches"] = list(used_values.keys())
        call["valid_mismatches"] = list(used_values.keys())
        return call
    vv = parsed.get("verified_values")
    call["verified_values"] = vv if isinstance(vv, dict) else {}
    mm = parsed.get("mismatches")
    call["mismatches"] = mm if isinstance(mm, list) else []
    valid, invalid, self_consistent = validate_mismatches(used_values, call["mismatches"], call["verified_values"])
    call["valid_mismatches"] = valid
    call["invalid_mismatch_items"] = invalid
    call["self_consistent_mismatch_items"] = self_consistent
    return call


def run_verifier_literate(client, model, table_text, question, used_values: dict, code: str):
    """Condition 3b's Verifier call: same value-mismatch handling as run_verifier,
    plus a logic_mismatch/logic_mismatch_reason judgment on Calculator's code
    (verifier_redesign_literate_pilot.md)."""
    user = VERIFIER_USER_TEMPLATE_LITERATE.format(
        table_as_text=table_text,
        question=question,
        calculator_used_values_json=json.dumps(used_values, ensure_ascii=False),
        calculator_code=code,
    )
    raw = call_llm(client, model, VERIFIER_SYSTEM_LITERATE, user)
    parsed, err = extract_json(raw)
    call = {
        "verified_values": {}, "value_mismatches": [], "valid_mismatches": [],
        "invalid_mismatch_items": [], "self_consistent_mismatch_items": [],
        "logic_mismatch": False, "logic_mismatch_reason": None,
        "raw_response": raw, "parse_error": err,
    }
    if parsed is None or not isinstance(parsed, dict):
        # Fail-safe: can't determine mismatches, so treat everything as unverified
        # (forces the retry / Fail path rather than silently accepting).
        call["value_mismatches"] = list(used_values.keys())
        call["valid_mismatches"] = list(used_values.keys())
        call["logic_mismatch"] = True
        call["logic_mismatch_reason"] = "verifier_response_unparseable"
        return call
    vv = parsed.get("verified_values")
    call["verified_values"] = vv if isinstance(vv, dict) else {}
    mm = parsed.get("value_mismatches")
    call["value_mismatches"] = mm if isinstance(mm, list) else []
    valid, invalid, self_consistent = validate_mismatches(used_values, call["value_mismatches"], call["verified_values"])
    call["valid_mismatches"] = valid
    call["invalid_mismatch_items"] = invalid
    call["self_consistent_mismatch_items"] = self_consistent
    call["logic_mismatch"] = bool(parsed.get("logic_mismatch"))
    call["logic_mismatch_reason"] = parsed.get("logic_mismatch_reason")
    return call


def build_literate_retry_feedback(used_values: dict, valid_mismatches: list, logic_mismatch: bool, logic_mismatch_reason) -> str:
    """Option D (verifier_redesign_literate_pilot.md): keep value-mismatch and
    logic-mismatch feedback distinguishable in the retry prompt, so the
    Calculator can tell whether it misread a cell vs. picked the wrong formula."""
    lines = []
    for item in valid_mismatches:
        val = used_values.get(item, "?")
        lines.append(f"- 값 불일치: {item}, 이전에 사용한 값 {val}, 표를 재확인하세요.")
    if logic_mismatch:
        reason = logic_mismatch_reason or "질문 의도와 code가 실제로 수행한 계산이 다릅니다."
        lines.append(f"- 계산 로직 문제: {reason}")
    return CALCULATOR_RETRY_FEEDBACK_TEMPLATE_LITERATE.format(issue_lines="\n".join(lines))


def em_match(value, gold_q: dict) -> bool:
    """Whether `value` matches gold_q's answer under the official TAT-QA EM metric
    (same scoring path as score_record) -- used to judge calculator_call_1's own
    correctness, independent of what the pipeline ultimately predicts."""
    if value is None:
        return False
    m = TaTQAEmAndF1()
    m(ground_truth=gold_q, prediction=str(value), pred_scale=gold_q.get("scale", ""))
    return m.get_raw()[-1]["em"] == 1.0


# ---------------------------------------------------------------------------
# Record builder + condition runners
# ---------------------------------------------------------------------------
def build_record(doc, q, condition, model):
    return {
        "gold": q,
        "table_uid": doc["table"]["uid"],
        "table": doc["table"]["table"],
        "condition": str(condition),
        "model": model,
        "predicted_answer": None,
        "is_correct": False,
        "calculator_call_1": None,
        "verifier_call_1": None,
        "retried": False,
        "retry_skipped_invalid_mismatch": False,
        "calculator_call_2": None,
        "verifier_call_2": None,
        "latency_seconds": 0.0,
        "timestamp": None,
    }


def finalize(rec, t0):
    rec["latency_seconds"] = time.perf_counter() - t0
    rec["timestamp"] = datetime.now(timezone.utc).isoformat()
    return rec


def run_condition1(client, model, doc, q):
    t0 = time.perf_counter()
    table_text = format_table(doc["table"])
    user = STANDARD_USER_TEMPLATE.format(table_as_text=table_text, question=q["question"])
    raw = call_llm(client, model, STANDARD_SYSTEM, user)
    parsed, err = extract_json(raw)
    rec = build_record(doc, q, 1, model)
    rec["calculator_call_1"] = {
        "used_values": {}, "code": "", "raw_response": raw,
        "code_exec_result": None, "code_exec_error": None, "parse_error": err,
        "internal_inconsistency": False, "unused_keys": [],
    }
    if parsed is not None and isinstance(parsed, dict) and "answer" in parsed:
        rec["predicted_answer"] = parsed["answer"]
    return finalize(rec, t0)


def run_shared_calculator(client, model, table_text, question):
    """Fix 1 (poc_fixes_v3.md): Condition 2 and Condition 3 must be built from the
    SAME calculator_call_1 -- one API call per question, not one each -- so that
    the two conditions' EM difference reflects only the Verifier's effect and not
    non-deterministic API sampling noise (temperature=0 is not fully deterministic;
    confirmed in the earlier gpt-5.4-mini diagnostic)."""
    t0 = time.perf_counter()
    calc1 = run_calculator(client, model, table_text, question)
    return calc1, time.perf_counter() - t0


def build_condition2_record(doc, q, model, calc1, calc1_latency):
    rec = build_record(doc, q, 2, model)
    rec["calculator_call_1"] = calc1
    if calc1["code_exec_error"] is None and not calc1["internal_inconsistency"]:
        rec["predicted_answer"] = calc1["code_exec_result"]
    rec["latency_seconds"] = calc1_latency
    rec["timestamp"] = datetime.now(timezone.utc).isoformat()
    return rec


def build_condition3_record(client, model, doc, q, table_text, calc1, calc1_latency):
    t_extra_start = time.perf_counter()
    rec = build_record(doc, q, 3, model)
    rec["calculator_call_1"] = calc1

    def done():
        rec["latency_seconds"] = calc1_latency + (time.perf_counter() - t_extra_start)
        rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        return rec

    if calc1["code_exec_error"] is not None:
        return done()  # exec/parse error -> Fail, no retry (spec: separate from verifier retry)
    if calc1["internal_inconsistency"]:
        return done()  # code doesn't use its own used_values -> Fail, no point asking Verifier

    verif1 = run_verifier(client, model, table_text, calc1["used_values"])
    rec["verifier_call_1"] = verif1

    if not verif1["mismatches"]:
        rec["predicted_answer"] = calc1["code_exec_result"]
        return done()

    if not verif1["valid_mismatches"]:
        # Fix 2: every claimed mismatch is a hallucinated/unrelated item name that
        # doesn't correspond to anything calc1 actually used -- don't retry on it.
        rec["retry_skipped_invalid_mismatch"] = True
        rec["predicted_answer"] = calc1["code_exec_result"]
        return done()

    rec["retried"] = True
    lines = []
    for item in verif1["valid_mismatches"]:
        val = calc1["used_values"].get(item, "?")
        lines.append(f"- {item}: 이전에 사용한 값 {val}, 표를 재확인하세요.")
    feedback = CALCULATOR_RETRY_FEEDBACK_TEMPLATE.format(mismatch_lines="\n".join(lines))

    calc2 = run_calculator(client, model, table_text, q["question"], feedback=feedback)
    rec["calculator_call_2"] = calc2
    if calc2["code_exec_error"] is not None:
        return done()
    if calc2["internal_inconsistency"]:
        return done()

    verif2 = run_verifier(client, model, table_text, calc2["used_values"])
    rec["verifier_call_2"] = verif2
    if not verif2["valid_mismatches"]:
        rec["predicted_answer"] = calc2["code_exec_result"]
    # else: still validly mismatched after retry -> Fail (predicted_answer stays None)
    return done()


def build_condition3b_record(client, model, doc, q, table_text, calc1, calc1_latency):
    """Condition 3b (열람형 Verifier, verifier_redesign_literate_pilot.md): same
    retry structure as Condition 3a (build_condition3_record) but the Verifier
    also sees Calculator's code/question and can flag a logic_mismatch, and the
    retry feedback (Option D) distinguishes value vs. logic issues."""
    t_extra_start = time.perf_counter()
    rec = build_record(doc, q, "3b", model)
    rec["calculator_call_1"] = calc1

    def done():
        rec["latency_seconds"] = calc1_latency + (time.perf_counter() - t_extra_start)
        rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        return rec

    if calc1["code_exec_error"] is not None:
        return done()  # exec/parse error -> Fail, no retry (spec: separate from verifier retry)
    if calc1["internal_inconsistency"]:
        return done()  # code doesn't use its own used_values -> Fail, no point asking Verifier

    verif1 = run_verifier_literate(client, model, table_text, q["question"], calc1["used_values"], calc1["code"])
    rec["verifier_call_1"] = verif1

    if not verif1["value_mismatches"] and not verif1["logic_mismatch"]:
        rec["predicted_answer"] = calc1["code_exec_result"]
        return done()

    if not verif1["valid_mismatches"] and not verif1["logic_mismatch"]:
        # Fix 2 (carried over from 3a): every claimed value mismatch is a
        # hallucinated/unrelated item name -- and no logic issue either -- so
        # don't retry on it.
        rec["retry_skipped_invalid_mismatch"] = True
        rec["predicted_answer"] = calc1["code_exec_result"]
        return done()

    rec["retried"] = True
    feedback = build_literate_retry_feedback(
        calc1["used_values"], verif1["valid_mismatches"], verif1["logic_mismatch"], verif1["logic_mismatch_reason"],
    )

    calc2 = run_calculator(client, model, table_text, q["question"], feedback=feedback)
    rec["calculator_call_2"] = calc2
    if calc2["code_exec_error"] is not None:
        return done()
    if calc2["internal_inconsistency"]:
        return done()

    verif2 = run_verifier_literate(client, model, table_text, q["question"], calc2["used_values"], calc2["code"])
    rec["verifier_call_2"] = verif2
    if not verif2["valid_mismatches"] and not verif2["logic_mismatch"]:
        rec["predicted_answer"] = calc2["code_exec_result"]
    # else: still validly mismatched or logic-mismatched after retry -> Fail
    return done()


# ---------------------------------------------------------------------------
# Official EM scoring (TAT-QA tatqa_metric.py, unmodified)
# ---------------------------------------------------------------------------
def score_record(meter: TaTQAEmAndF1, rec: dict, gold_q: dict):
    pred = rec["predicted_answer"]
    pred_str = None if pred is None else str(pred)
    meter(ground_truth=gold_q, prediction=pred_str, pred_scale=gold_q.get("scale", ""))
    is_correct = meter.get_raw()[-1]["em"] == 1.0
    rec["is_correct"] = bool(is_correct)


# ---------------------------------------------------------------------------
# Verifier oracle (per verifier_oracle_spec.md)
# ---------------------------------------------------------------------------
def find_matching_rows(table_rows, item_name):
    """Exact normalized-label matches always count. A substring match only counts
    if the row actually has data -- otherwise a generic section header (e.g. "Sales:")
    that happens to share a short substring with a fully-spelled item name would
    falsely stand in for the real (unmatched, e.g. due to a table typo) row and
    make oracle_judge call a correct value "incorrect" against a blank row.

    Rows whose first column (label) is blank are subtotal/total rows in practice
    (a labeled subsection followed by an unlabeled sum), and item_name commonly
    marks these with a trailing "(Total)" segment (see CALCULATOR prompt
    convention). Such a row is included as a candidate only when item_name wants
    a total AND it immediately follows another data row -- i.e. it plausibly
    closes a block of labeled subrows, not a stray blank row.

    Returns (exact, substring, total_candidates) as three separate lists of
    (row_index, row) tuples, so callers can treat an exact label match as
    authoritative (fix_oracle_judge_rescore.md) -- an exact "Total current tax
    expense" row must win over an unrelated unlabeled subtotal row that also
    happens to satisfy the weaker "wants a total" heuristic, not just be tried
    alongside it. Exact/substring comparison uses item_name with its trailing
    "(<column token>)" segment stripped off (e.g. "Foo (2019)" -> "Foo"),
    since that trailing segment is the Calculator's own column marker, not part
    of the row's label -- comparing the full item_name against row labels would
    make an exact match essentially never fire (row labels never carry a "(2019)"
    suffix in this dataset)."""
    base_label = re.sub(r"\s*\([^)]*\)\s*$", "", item_name).strip()
    norm_item_full = normalize_label(item_name)
    norm_item = normalize_label(base_label) or norm_item_full
    if not norm_item:
        return [], [], []
    wants_total = "total" in norm_item_full
    exact, substring, total_candidates = [], [], []
    for idx, row in enumerate(table_rows):
        if not row:
            continue
        norm_label = normalize_label(row[0])
        has_data = any(str(c).strip() != "" for c in row[1:])
        if norm_label:
            if norm_item == norm_label:
                exact.append((idx, row))
            elif norm_item in norm_label or norm_label in norm_item:
                if has_data:
                    substring.append((idx, row))
        elif wants_total and has_data and idx > 0:
            prev_row = table_rows[idx - 1]
            if prev_row and normalize_label(prev_row[0]):
                total_candidates.append((idx, row))
    return exact, substring, total_candidates


def build_column_headers(table_rows, row_idx):
    """Combined, forward-filled header text per column, built from all rows
    above row_idx (0..row_idx-1). Forward-fill (within a row, left to right)
    models merged header cells: a table with a top row ["", "", "Net additions
    (losses)", "", "% of penetration", ""] and a sub-header row ["", "Aug 2019",
    "Aug 2019", "Aug 2018", ...] means column 2's real header is "Net additions
    (losses) Aug 2019", not just "Aug 2019" -- the group label in row 0 is blank
    in column 3 only because it visually spans columns 2-3 (fix_oracle_judge_rescore.md)."""
    header_rows = table_rows[:row_idx]
    if not header_rows:
        return []
    ncols = max((len(r) for r in header_rows), default=0)
    combined = [""] * ncols
    for row in header_rows:
        last_val = ""
        for col_idx in range(ncols):
            cell = str(row[col_idx]).strip() if col_idx < len(row) else ""
            if cell:
                last_val = cell
            if last_val:
                combined[col_idx] = (combined[col_idx] + " " + last_val).strip()
    return combined


def find_header_column(table_rows, row_idx, item_name):
    """Find the column (index >= 1) that item_name's trailing "(<token>)" refers
    to. A hierarchical header (e.g. Domestic/International groups each with
    their own 2019/2018 sub-columns) can have the SAME token (e.g. "2019")
    appear verbatim in more than one column; in that case, disambiguate using
    ALL of item_name's parenthetical segments against the combined, forward-
    filled header text for each candidate column (so "Net additions (losses)
    (August 31, 2019)" picks the "Net additions (losses)" group's Aug-2019
    column, not an unrelated column that also happens to say "August 31, 2019").
    Returns None (safe fallback to a full-row scan) rather than guess when
    still ambiguous after that (fix_oracle_judge_rescore.md)."""
    token = parse_item_column_token(item_name)
    if not token:
        return None
    norm_target = normalize_label(token)
    if not norm_target:
        return None
    candidates = set()
    for idx in range(row_idx - 1, -1, -1):
        row = table_rows[idx]
        if not row:
            continue
        for col_idx, cell in enumerate(row):
            if col_idx == 0:
                continue
            if normalize_label(cell) == norm_target:
                candidates.add(col_idx)
    if not candidates:
        return None
    if len(candidates) == 1:
        return next(iter(candidates))

    all_tokens = [normalize_label(g) for g in re.findall(r"\(([^)]*)\)", item_name)]
    all_tokens = [t for t in all_tokens if t]
    combined = build_column_headers(table_rows, row_idx)
    matches = [c for c in sorted(candidates)
               if c < len(combined) and all(t in normalize_label(combined[c]) for t in all_tokens)]
    return matches[0] if len(matches) == 1 else None


def parse_item_column_token(item_name):
    """Last parenthetical segment of item_name is the Calculator's own marker for
    which column/time-period it read the value from (e.g. "... (2018)", "...
    (Payments)") -- not necessarily a year. Used to align claimed_value against
    the correct table column instead of any cell in the matched row."""
    groups = re.findall(r"\(([^)]*)\)", item_name)
    return groups[-1].strip() if groups and groups[-1].strip() else None


def _oracle_judge_over(table_rows, candidates, item_name, claimed_num):
    """Try each (row_index, row) candidate; return ("correct", row) on the first
    match, else (None, first_row) if none of them match."""
    for row_idx, row in candidates:
        col_idx = find_header_column(table_rows, row_idx, item_name)
        if col_idx is not None and col_idx < len(row):
            cell_num = oracle_parse_number(row[col_idx])
            if cell_num is not None:
                if abs(cell_num - claimed_num) < 1e-6:
                    return "correct", row
                continue  # aligned column found and it disagrees -- try next candidate row
        # No usable column alignment for this row: fall back to any-cell scan.
        for cell in row[1:]:
            cell_num = oracle_parse_number(cell)
            if cell_num is not None and abs(cell_num - claimed_num) < 1e-6:
                return "correct", row
    return None, candidates[0][1]


def oracle_judge(table_rows, item_name, claimed_value):
    exact, substring, total_candidates = find_matching_rows(table_rows, item_name)
    if not (exact or substring or total_candidates):
        return "ambiguous", None
    claimed_num = oracle_parse_number(claimed_value)
    if claimed_num is None:
        return "ambiguous", (exact or substring or total_candidates)[0][1]

    if exact:
        # An exact label match is authoritative: don't let a weaker substring/
        # total-row candidate override a definitive column-aligned mismatch here.
        verdict, row = _oracle_judge_over(table_rows, exact, item_name, claimed_num)
        return (verdict or "incorrect"), row

    verdict, row = _oracle_judge_over(table_rows, substring + total_candidates, item_name, claimed_num)
    return (verdict or "incorrect"), row


def mismatch_flagged(item_name, mismatches):
    norm_item = normalize_label(item_name)
    return any(normalize_label(m) == norm_item for m in mismatches)


def verify_shared_calculator_consistency(short, tag_suffix, model, cond3_label="3"):
    """Fix 1 sanity check: condition2 and condition3(/3b)'s calculator_call_1 must
    be byte-for-byte the same record for every question, since both are now built
    from one shared Calculator call. Raises AssertionError if they ever diverge."""
    path2 = RESULTS_DIR / f"poc_{short}_condition2{tag_suffix}.jsonl"
    path3 = RESULTS_DIR / f"poc_{short}_condition{cond3_label}{tag_suffix}.jsonl"
    if not (path2.exists() and path3.exists()):
        print(f"[shared-calculator check] model={model}: skipped (condition2/{cond3_label} file missing)")
        return

    def load(path):
        out = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out[d["gold"]["uid"]] = d["calculator_call_1"]
        return out

    m2, m3 = load(path2), load(path3)
    mismatched = []
    for uid, c2 in m2.items():
        c3 = m3.get(uid)
        if c3 is None:
            mismatched.append((uid, f"missing_in_condition{cond3_label}"))
        elif c2.get("code") != c3.get("code") or c2.get("used_values") != c3.get("used_values"):
            mismatched.append((uid, "code_or_used_values_differ"))

    ok = not mismatched
    print(f"[shared-calculator check] model={model} (condition2 vs {cond3_label}): "
          f"{'ALL MATCH' if ok else str(len(mismatched)) + ' MISMATCHES'} "
          f"({len(m2)} questions compared)")
    if not ok:
        for uid, reason in mismatched[:10]:
            print(f"    MISMATCH uid={uid} ({reason})")
    assert ok, f"calculator_call_1 mismatch between condition2/condition{cond3_label} for model={model}: {mismatched}"


def process_verifier_for_audit(state, model, condition, q_uid, table_uid, table_rows, calc_call, verif_call, call_idx):
    for item_name, claimed_value in calc_call["used_values"].items():
        verdict, matched_row = oracle_judge(table_rows, item_name, claimed_value)
        mismatch_names = verif_call.get("mismatches", verif_call.get("value_mismatches", []))
        flagged = mismatch_flagged(item_name, mismatch_names)
        audit_rec = {
            "model": model,
            "question_uid": q_uid,
            "table_uid": table_uid,
            "verifier_call": call_idx,
            "item_name": item_name,
            "claimed_value": claimed_value,
            "matched_row": matched_row,
            "verdict": verdict,
            "verifier_flagged_mismatch": flagged,
            "table": table_rows,
        }
        stats = state["oracle_stats"][(model, condition)]
        if verdict == "ambiguous":
            stats["n_ambiguous"] += 1
            state["audit_ambiguous"].append(audit_rec)
            continue
        state["audit_pool"].append(audit_rec)
        is_wrong = verdict == "incorrect"
        if is_wrong and flagged:
            stats["tp"] += 1
        elif (not is_wrong) and flagged:
            stats["fp"] += 1
        elif is_wrong and not flagged:
            stats["fn"] += 1
        else:
            stats["tn"] += 1


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit number of questions (smoke test)")
    parser.add_argument("--start", type=int, default=0, help="slice start index into poc_sample.json (for chunked/resumed runs)")
    parser.add_argument("--end", type=int, default=None, help="slice end index into poc_sample.json (for chunked/resumed runs)")
    parser.add_argument("--append", action="store_true", help="append to existing per-condition jsonl instead of overwriting (use for later chunks of a split run)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="subset of models to run")
    parser.add_argument("--conditions", nargs="+", type=str, default=["1", "2", "3"],
                         help="subset of conditions to run: 1 (Standard), 2 (PoT), "
                              "3 (Proposed, blind Verifier), 3b (Proposed, literate Verifier -- "
                              "sees Calculator's code, see verifier_redesign_literate_pilot.md)")
    parser.add_argument("--no-summary", action="store_true", help="skip writing poc_summary.json/audit_sample.jsonl (use build_summary.py once all chunks are done)")
    parser.add_argument("--tag", default="", help="suffix appended to all output filenames, e.g. --tag v2 -> poc_qwen_condition2_v2.jsonl / poc_summary_v2.json")
    parser.add_argument("--sample", default=str(SAMPLE_PATH), help="path to sample json file (default: filtered_tatqa/poc_sample.json)")
    args = parser.parse_args()
    tag_suffix = f"_{args.tag}" if args.tag else ""
    sample_path = Path(args.sample)

    with sample_path.open(encoding="utf-8") as f:
        sample_docs = json.load(f)
    sample_docs = sample_docs[args.start: args.end]
    if args.limit:
        sample_docs = sample_docs[: args.limit]

    RESULTS_DIR.mkdir(exist_ok=True)
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    state = {
        "oracle_stats": {(m, c): {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n_ambiguous": 0}
                          for m in args.models for c in args.conditions},
        "audit_pool": [],
        "audit_ambiguous": [],
    }
    meters = {}
    latency_by_mc = {}
    retry_stats = {(m, c): {
        "n_retried": 0,
        "n_retry_truly_fixed": 0,
        "n_retry_regressed": 0,
        "n_retry_verifier_satisfied_but_still_wrong": 0,
        "n_retry_invalid_mismatch": 0,
        "n_logic_mismatch_flagged": 0,
    } for m in args.models for c in args.conditions}
    inconsistency_stats = {m: {c: 0 for c in args.conditions} for m in args.models}
    file_mode = "a" if args.append else "w"
    needs_shared_calc = any(c in ("2", "3", "3b") for c in args.conditions)

    for model in args.models:
        short = MODELS[model]
        print(f"\n=== Model: {model} ===")
        for condition in args.conditions:
            meters[(model, condition)] = TaTQAEmAndF1()
            latency_by_mc[(model, condition)] = []
        n_correct_running = {c: 0 for c in args.conditions}

        file_handles = {
            c: (RESULTS_DIR / f"poc_{short}_condition{c}{tag_suffix}.jsonl").open(file_mode, encoding="utf-8")
            for c in args.conditions
        }
        try:
            for i, doc in enumerate(sample_docs, 1):
                q = doc["questions"][0]
                table_text = format_table(doc["table"])

                # Fix 1: one shared calculator_call_1 per question, reused by both
                # condition 2 (its final answer) and condition 3 (fed to Verifier).
                shared_calc1, shared_calc1_latency = (
                    run_shared_calculator(client, model, table_text, q["question"])
                    if needs_shared_calc else (None, 0.0)
                )

                for condition in args.conditions:
                    if condition == "1":
                        rec = run_condition1(client, model, doc, q)
                    elif condition == "2":
                        rec = build_condition2_record(doc, q, model, shared_calc1, shared_calc1_latency)
                    else:
                        builder = build_condition3b_record if condition == "3b" else build_condition3_record
                        rec = builder(
                            client, model, doc, q, table_text, shared_calc1, shared_calc1_latency,
                        )
                        if rec["verifier_call_1"] is not None:
                            process_verifier_for_audit(
                                state, model, condition, q["uid"], doc["table"]["uid"], doc["table"]["table"],
                                rec["calculator_call_1"], rec["verifier_call_1"], 1,
                            )
                        if rec["retry_skipped_invalid_mismatch"]:
                            retry_stats[(model, condition)]["n_retry_invalid_mismatch"] += 1
                        if rec["retried"] and rec["verifier_call_2"] is not None:
                            process_verifier_for_audit(
                                state, model, condition, q["uid"], doc["table"]["uid"], doc["table"]["table"],
                                rec["calculator_call_2"], rec["verifier_call_2"], 2,
                            )

                    if condition in ("2", "3", "3b"):
                        calc1_inc = bool(rec["calculator_call_1"] and rec["calculator_call_1"]["internal_inconsistency"])
                        calc2_inc = bool(rec.get("calculator_call_2") and rec["calculator_call_2"]["internal_inconsistency"])
                        if calc1_inc or calc2_inc:
                            inconsistency_stats[model][condition] += 1

                    score_record(meters[(model, condition)], rec, q)

                    # Fix 3: redefined retry-effect metrics, computed only now that
                    # rec["is_correct"] (final) has been scored above.
                    if condition in ("3", "3b") and rec["retried"]:
                        rs = retry_stats[(model, condition)]
                        rs["n_retried"] += 1
                        calc1_correct = em_match(rec["calculator_call_1"]["code_exec_result"], q)
                        final_correct = rec["is_correct"]
                        if not calc1_correct and final_correct:
                            rs["n_retry_truly_fixed"] += 1
                        if calc1_correct and not final_correct:
                            rs["n_retry_regressed"] += 1
                        verifier_satisfied = rec["predicted_answer"] is not None
                        if verifier_satisfied and not final_correct:
                            rs["n_retry_verifier_satisfied_but_still_wrong"] += 1
                        if condition == "3b" and rec["verifier_call_1"].get("logic_mismatch"):
                            rs["n_logic_mismatch_flagged"] += 1

                    if rec["is_correct"]:
                        n_correct_running[condition] += 1
                    latency_by_mc[(model, condition)].append(rec["latency_seconds"])

                    fout = file_handles[condition]
                    fout.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fout.flush()
                    print(f"  [cond{condition}] {i}/{len(sample_docs)} uid={q['uid']} "
                          f"correct={rec['is_correct']} ({n_correct_running[condition]}/{i}) "
                          f"t={rec['latency_seconds']:.2f}s")
        finally:
            for fout in file_handles.values():
                fout.close()

        if "2" in args.conditions and "3" in args.conditions:
            verify_shared_calculator_consistency(short, tag_suffix, model, cond3_label="3")
        if "2" in args.conditions and "3b" in args.conditions:
            verify_shared_calculator_consistency(short, tag_suffix, model, cond3_label="3b")

    if args.no_summary:
        print("\n--no-summary set: skipping poc_summary.json/audit_sample.jsonl "
              "(run build_summary.py once all chunks/models/conditions are done)")
        return

    # ---- Summary ----
    summary = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(sample_docs),
        "results": {},
    }
    for model in args.models:
        model_summary = {}
        for condition in args.conditions:
            meter = meters[(model, condition)]
            em, _f1, _scale_score, _op_score = meter.get_overall_metric()
            raw = meter.get_raw()
            n_total = len(raw)
            n_correct = sum(1 for d in raw if d["em"] == 1.0)
            cond_summary = {"em": em, "n_correct": n_correct, "n_total": n_total}
            if condition in ("2", "3", "3b"):
                cond_summary["n_internal_inconsistency"] = inconsistency_stats[model][condition]
            if condition in ("3", "3b"):
                st = state["oracle_stats"][(model, condition)]
                precision = st["tp"] / (st["tp"] + st["fp"]) if (st["tp"] + st["fp"]) > 0 else None
                recall = st["tp"] / (st["tp"] + st["fn"]) if (st["tp"] + st["fn"]) > 0 else None
                lat = latency_by_mc[(model, condition)]
                rs = retry_stats[(model, condition)]
                cond_summary.update({
                    "verifier_precision": precision,
                    "verifier_recall": recall,
                    "n_ambiguous": st["n_ambiguous"],
                    "automated_not_manually_validated": True,
                    "n_retried": rs["n_retried"],
                    "n_retry_truly_fixed": rs["n_retry_truly_fixed"],
                    "n_retry_regressed": rs["n_retry_regressed"],
                    "n_retry_verifier_satisfied_but_still_wrong":
                        rs["n_retry_verifier_satisfied_but_still_wrong"],
                    "n_retry_invalid_mismatch": rs["n_retry_invalid_mismatch"],
                    "avg_latency_seconds": (sum(lat) / len(lat)) if lat else 0.0,
                })
                if condition == "3b":
                    cond_summary["n_logic_mismatch_flagged"] = rs["n_logic_mismatch_flagged"]
            model_summary[f"condition{condition}"] = cond_summary
        summary["results"][model] = model_summary

    with (RESULTS_DIR / f"poc_summary{tag_suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- Verifier audit sample (verifier_oracle_spec.md) ----
    rng = random.Random(11)
    pool = state["audit_pool"]
    sampled = rng.sample(pool, min(20, len(pool))) if pool else []
    audit_records = state["audit_ambiguous"] + sampled
    with (RESULTS_DIR / f"audit_sample{tag_suffix}.jsonl").open("w", encoding="utf-8") as f:
        for rec in audit_records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  summary -> {RESULTS_DIR / f'poc_summary{tag_suffix}.json'}")
    print(f"  audit sample ({len(audit_records)} rows) -> {RESULTS_DIR / f'audit_sample{tag_suffix}.jsonl'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
