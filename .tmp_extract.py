import json
from pathlib import Path

base = Path('results')
models = ['qwen3-1.7b','qwen3-4b','qwen3-8b']
conds = ['3','3b']

out = {}
for m in models:
    out[m] = {}
    for c in conds:
        p = base / f'poc_{m}_condition{c}_clean1000.jsonl'
        if not p.exists():
            continue
        n=0
        n_correct=0
        n_retried=0
        n_fix=0
        n_reg=0
        tp=fp=fn=tn=amb=0
        for line in p.open(encoding='utf-8'):
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if r.get('is_correct'): n_correct += 1
            if r.get('retried'): n_retried += 1
            n_fix += int(r.get('retried') and r.get('is_correct') and not r.get('calculator_call_1',{}).get('is_correct', False)) if False else 0
            n_fix = n_fix
            if r.get('retried'):
                # derived exactly like logged stats using available booleans
                calc1 = r.get('calculator_call_1', {})
                # use precomputed counters from records when available at summary level only; keep placeholders not used
                pass

            for call_k, calc_k in [('verifier_call_1','calculator_call_1'), ('verifier_call_2','calculator_call_2')]:
                ver = r.get(call_k)
                calc = r.get(calc_k)
                if not ver or not calc:
                    continue
                used = calc.get('used_values', {}) or {}
                mism = ver.get('mismatches', ver.get('value_mismatches', [])) or []
                verified = ver.get('verified_values', {}) or {}
                # valid mismatch approx per pipeline: in used and value differs
                valid = []
                for k in mism:
                    if k in used and (k not in verified or verified.get(k) != used.get(k)):
                        valid.append(k)
                for item, claimed in used.items():
                    if item not in verified:
                        amb += 1
                        continue
                    wrong = verified.get(item) != claimed
                    flagged = item in valid
                    if wrong and flagged: tp += 1
                    elif (not wrong) and flagged: fp += 1
                    elif wrong and (not flagged): fn += 1
                    else: tn += 1

        prec = tp/(tp+fp) if (tp+fp)>0 else None
        rec = tp/(tp+fn) if (tp+fn)>0 else None
        out[m][c] = {
            'em': n_correct/n if n else None,
            'n_total': n,
            'n_correct': n_correct,
            'precision_approx': prec,
            'recall_approx': rec,
            'n_ambiguous_approx': amb,
            'n_retried': n_retried,
        }

print(json.dumps(out, ensure_ascii=False, indent=2))
