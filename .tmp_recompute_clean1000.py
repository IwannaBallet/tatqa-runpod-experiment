import json
from pathlib import Path
import run_poc

base = Path('results')
models = ['qwen3-1.7b','qwen3-4b','qwen3-8b']
conds = ['3','3b']

# Map filename token to model id used in logs
model_id = {
    'qwen3-1.7b': 'qwen3:1.7b',
    'qwen3-4b': 'qwen3:4b',
    'qwen3-8b': 'qwen3:8b',
}

summary = {}

for m in models:
    summary[m] = {'pot_from_calc1': None, 'conditions': {}}

    # PoT from shared calculator_call_1 in condition3 clean1000
    c3_path = base / f'poc_{m}_condition3_clean1000.jsonl'
    if c3_path.exists():
        n=0; n_ok=0
        with c3_path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                if run_poc.em_match(r['calculator_call_1'].get('code_exec_result'), r['gold']):
                    n_ok += 1
        summary[m]['pot_from_calc1'] = {'em': n_ok/n if n else None, 'n_correct': n_ok, 'n_total': n}

    for c in conds:
        p = base / f'poc_{m}_condition{c}_clean1000.jsonl'
        if not p.exists():
            continue
        records = []
        with p.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        n_total = len(records)
        n_correct = sum(1 for r in records if r.get('is_correct'))

        state = {
            'oracle_stats': {(model_id[m], c): {'tp':0,'fp':0,'fn':0,'tn':0,'n_ambiguous':0}},
            'audit_pool': [],
            'audit_ambiguous': [],
        }

        n_retried=0
        n_retry_truly_fixed=0
        n_retry_regressed=0

        for r in records:
            if r.get('verifier_call_1') is not None:
                run_poc.process_verifier_for_audit(
                    state, model_id[m], c, r['gold']['uid'], r['table_uid'], r['table'],
                    r['calculator_call_1'], r['verifier_call_1'], 1
                )
            if r.get('retried') and r.get('verifier_call_2') is not None and r.get('calculator_call_2') is not None:
                run_poc.process_verifier_for_audit(
                    state, model_id[m], c, r['gold']['uid'], r['table_uid'], r['table'],
                    r['calculator_call_2'], r['verifier_call_2'], 2
                )

            if r.get('retried'):
                n_retried += 1
                calc1_correct = run_poc.em_match(r['calculator_call_1'].get('code_exec_result'), r['gold'])
                final_correct = bool(r.get('is_correct'))
                if (not calc1_correct) and final_correct:
                    n_retry_truly_fixed += 1
                if calc1_correct and (not final_correct):
                    n_retry_regressed += 1

        st = state['oracle_stats'][(model_id[m], c)]
        precision = st['tp'] / (st['tp'] + st['fp']) if (st['tp'] + st['fp']) > 0 else None
        recall = st['tp'] / (st['tp'] + st['fn']) if (st['tp'] + st['fn']) > 0 else None

        summary[m]['conditions'][c] = {
            'em': n_correct / n_total if n_total else None,
            'n_correct': n_correct,
            'n_total': n_total,
            'precision': precision,
            'recall': recall,
            'n_retried': n_retried,
            'n_retry_truly_fixed': n_retry_truly_fixed,
            'n_retry_regressed': n_retry_regressed,
        }

print(json.dumps(summary, ensure_ascii=False, indent=2))
