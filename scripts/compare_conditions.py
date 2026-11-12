import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'results'
summary_file = p / 'poc_summary_clean1000.json'
file_c3 = p / 'poc_qwen3-1.7b_condition3_clean1000.jsonl'
file_c3b = p / 'poc_qwen3-1.7b_condition3b_clean1000.jsonl'

with open(summary_file, 'r', encoding='utf-8') as f:
    summary = json.load(f)

res = summary['results']['qwen3:1.7b']
metrics = [
    ('EM', 'em'),
    ('verifier_precision', 'verifier_precision'),
    ('verifier_recall', 'verifier_recall'),
    ('n_retried', 'n_retried'),
    ('n_retry_truly_fixed', 'n_retry_truly_fixed'),
    ('n_retry_regressed', 'n_retry_regressed'),
    ('n_retry_verifier_satisfied_but_still_wrong', 'n_retry_verifier_satisfied_but_still_wrong'),
]

print('Comparison for qwen3:1.7b — condition3 vs condition3b')
print('Metric\tcondition3\tcondition3b')
for label, key in metrics:
    v1 = res['condition3'].get(key, None)
    v2 = res['condition3b'].get(key, None)
    print(f"{label}\t{v1}\t{v2}")

# Load jsonl

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
    # Some files use 'retried' key (bool) or 'n_retried' elsewhere. Use top-level 'retried' if present, else check calculator_call_2 not None
    retried_a = a.get('retried', False)
    retried_b = b.get('retried', False)
    if (not retried_a) and (not retried_b):
        both_no_retry.append((uid, a, b))

n_both = len(both_no_retry)
print('\nItems where retried == false in BOTH runs: {} (of {})'.format(n_both, len(common_uids)))

if n_both == 0:
    print('No matching items without retries found.')
    raise SystemExit(0)

# Compute EM in subset
em_a = sum(1 for (_uid,a,b) in both_no_retry if a.get('is_correct')) / n_both
em_b = sum(1 for (_uid,a,b) in both_no_retry if b.get('is_correct')) / n_both
print(f"\nExact Match (EM) within no-retry subset — condition3: {em_a:.4f}, condition3b: {em_b:.4f}")

# Analyze differences causes

def safe_get(item, *path, default=None):
    cur = item
    for p_ in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p_, default)
        else:
            return default
    return cur

counts = {
    'internal_inconsistency_a': 0,
    'internal_inconsistency_b': 0,
    'internal_inconsistency_diff': 0,
    'api_timeout_a': 0,
    'api_timeout_b': 0,
    'api_timeout_diff': 0,
    'verifier_mismatch_a': 0,
    'verifier_mismatch_b': 0,
    'verifier_mismatch_diff': 0,
}

for (_uid,a,b) in both_no_retry:
    a_calc_in = safe_get(a, 'calculator_call_1', 'internal_inconsistency', default=False)
    b_calc_in = safe_get(b, 'calculator_call_1', 'internal_inconsistency', default=False)
    if a_calc_in:
        counts['internal_inconsistency_a'] += 1
    if b_calc_in:
        counts['internal_inconsistency_b'] += 1
    if bool(a_calc_in) != bool(b_calc_in):
        counts['internal_inconsistency_diff'] += 1

    a_api = bool(safe_get(a, 'calculator_call_1', 'api_timeout', default=False)) or bool(safe_get(a, 'verifier_call_1', 'api_timeout', default=False))
    b_api = bool(safe_get(b, 'calculator_call_1', 'api_timeout', default=False)) or bool(safe_get(b, 'verifier_call_1', 'api_timeout', default=False))
    if a_api:
        counts['api_timeout_a'] += 1
    if b_api:
        counts['api_timeout_b'] += 1
    if a_api != b_api:
        counts['api_timeout_diff'] += 1

    a_vm = len(safe_get(a, 'verifier_call_1', 'mismatches', default=[])) if safe_get(a, 'verifier_call_1') else 0
    b_vm = len(safe_get(b, 'verifier_call_1', 'mismatches', default=[])) if safe_get(b, 'verifier_call_1') else 0
    if a_vm>0:
        counts['verifier_mismatch_a'] += 1
    if b_vm>0:
        counts['verifier_mismatch_b'] += 1
    if (a_vm>0) != (b_vm>0):
        counts['verifier_mismatch_diff'] += 1

print('\nCounts in no-retry subset (per-run and diffs):')
for k,v in counts.items():
    print(f"{k}: {v}")

# If EMs differ significantly, list a small sample of uids where the answers differ and show reasons
if abs(em_a - em_b) > 0.01:
    print('\nSample diagnostics for items with differing correctness:')
    sample_count = 0
    for uid,a,b in both_no_retry:
        if a.get('is_correct') != b.get('is_correct'):
            sample_count += 1
            print('UID:', uid)
            print('  cond3 is_correct:', a.get('is_correct'))
            print('  cond3b is_correct:', b.get('is_correct'))
            print('  cond3 calculator internal_inconsistency:', safe_get(a, 'calculator_call_1', 'internal_inconsistency'))
            print('  cond3b calculator internal_inconsistency:', safe_get(b, 'calculator_call_1', 'internal_inconsistency'))
            print('  cond3 calculator api_timeout:', safe_get(a, 'calculator_call_1', 'api_timeout'))
            print('  cond3b calculator api_timeout:', safe_get(b, 'calculator_call_1', 'api_timeout'))
            print('  cond3 verifier mismatches:', safe_get(a, 'verifier_call_1', 'mismatches'))
            print('  cond3b verifier mismatches:', safe_get(b, 'verifier_call_1', 'mismatches'))
            print('')
            if sample_count >= 10:
                break

print('\nDone')
