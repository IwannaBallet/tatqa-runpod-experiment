import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'results'
summary_file = p / 'poc_summary_clean1000.json'
file_c3 = p / 'poc_qwen3-4b_condition3_clean1000.jsonl'
file_c3b = p / 'poc_qwen3-4b_condition3b_clean1000.jsonl'

with open(summary_file, 'r', encoding='utf-8') as f:
    summary = json.load(f)

res = summary['results'].get('qwen3:4b')
if not res:
    print('No qwen3:4b entry in summary')
    raise SystemExit(1)

metrics = [
    ('EM', 'em'),
    ('verifier_precision', 'verifier_precision'),
    ('verifier_recall', 'verifier_recall'),
    ('n_retried', 'n_retried'),
    ('n_retry_truly_fixed', 'n_retry_truly_fixed'),
    ('n_retry_regressed', 'n_retry_regressed'),
    ('n_retry_verifier_satisfied_but_still_wrong', 'n_retry_verifier_satisfied_but_still_wrong'),
]

print('qwen3:4b — condition3 vs condition3b')
print('Metric\tcondition3\tcondition3b')
for label, key in metrics:
    v1 = res['condition3'].get(key, None)
    v2 = res['condition3b'].get(key, None)
    print(f"{label}\t{v1}\t{v2}")

# load jsonl

def load_jsonl(path):
    arr = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                arr.append(json.loads(line))
    return arr

c3 = load_jsonl(file_c3)
c3b = load_jsonl(file_c3b)

by_uid_c3 = {item['gold']['uid']: item for item in c3}
by_uid_c3b = {item['gold']['uid']: item for item in c3b}

common_uids = set(by_uid_c3.keys()) & set(by_uid_c3b.keys())

both_no_retry = []
for uid in common_uids:
    a = by_uid_c3[uid]
    b = by_uid_c3b[uid]
    retried_a = a.get('retried', False)
    retried_b = b.get('retried', False)
    if (not retried_a) and (not retried_b):
        both_no_retry.append((uid, a, b))

n_both = len(both_no_retry)
print('\nItems where retried == false in BOTH runs: {} (of {})'.format(n_both, len(common_uids)))

if n_both == 0:
    print('No matching items without retries found for 4b.')
    raise SystemExit(0)

em_a = sum(1 for (_uid,a,b) in both_no_retry if a.get('is_correct')) / n_both
em_b = sum(1 for (_uid,a,b) in both_no_retry if b.get('is_correct')) / n_both
print(f"\nExact Match (EM) within no-retry subset — condition3: {em_a:.4f}, condition3b: {em_b:.4f}")

# Compare with qwen3:1.7b from summary if present
ref = summary['results'].get('qwen3:1.7b')
if ref:
    em_17_c3 = ref['condition3'].get('em')
    em_17_c3b = ref['condition3b'].get('em')
    gap_17 = (em_17_c3 - em_17_c3b) if (em_17_c3 is not None and em_17_c3b is not None) else None
    em_4b_c3 = res['condition3'].get('em')
    em_4b_c3b = res['condition3b'].get('em')
    gap_4b = (em_4b_c3 - em_4b_c3b) if (em_4b_c3 is not None and em_4b_c3b is not None) else None
    print('\nComparison with qwen3:1.7b:')
    print(f"1.7B EMs — condition3: {em_17_c3}, condition3b: {em_17_c3b}, gap: {gap_17}")
    print(f"4B EMs  — condition3: {em_4b_c3}, condition3b: {em_4b_c3b}, gap: {gap_4b}")
    if gap_17 is not None and gap_4b is not None:
        if (gap_4b > gap_17):
            trend = '격차가 커졌음'
        elif (gap_4b < gap_17):
            trend = '격차가 줄었음'
        else:
            trend = '격차가 같음'
        print('Trend vs 1.7B gap:', trend)

# Also compute some diagnostics in both_no_retry for internal_inconsistency and verifier mismatches
counts = {'internal_inconsistency_a':0,'internal_inconsistency_b':0,'internal_inconsistency_diff':0,
          'api_timeout_a':0,'api_timeout_b':0,'api_timeout_diff':0,
          'verifier_mismatch_a':0,'verifier_mismatch_b':0,'verifier_mismatch_diff':0}

for (_uid,a,b) in both_no_retry:
    a_calc_in = a.get('calculator_call_1', {}).get('internal_inconsistency', False)
    b_calc_in = b.get('calculator_call_1', {}).get('internal_inconsistency', False)
    if a_calc_in: counts['internal_inconsistency_a'] += 1
    if b_calc_in: counts['internal_inconsistency_b'] += 1
    if bool(a_calc_in) != bool(b_calc_in): counts['internal_inconsistency_diff'] += 1

    a_api = bool(a.get('calculator_call_1', {}).get('api_timeout')) or bool(a.get('verifier_call_1', {}).get('api_timeout') if a.get('verifier_call_1') else False)
    b_api = bool(b.get('calculator_call_1', {}).get('api_timeout')) or bool(b.get('verifier_call_1', {}).get('api_timeout') if b.get('verifier_call_1') else False)
    if a_api: counts['api_timeout_a'] += 1
    if b_api: counts['api_timeout_b'] += 1
    if a_api != b_api: counts['api_timeout_diff'] += 1

    a_vm = len(a.get('verifier_call_1', {}).get('mismatches', [])) if a.get('verifier_call_1') else 0
    b_vm = len(b.get('verifier_call_1', {}).get('mismatches', [])) if b.get('verifier_call_1') else 0
    if a_vm>0: counts['verifier_mismatch_a'] += 1
    if b_vm>0: counts['verifier_mismatch_b'] += 1
    if (a_vm>0) != (b_vm>0): counts['verifier_mismatch_diff'] += 1

print('\nDiagnostics in no-retry subset (per-run and diffs):')
for k,v in counts.items():
    print(f"{k}: {v}")

print('\nDone')
