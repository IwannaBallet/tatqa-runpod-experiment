# Calculator / Verifier 프롬프트 스펙

## 참고 문서
- 연구 배경: RESEARCH_CONTEXT.md (있다면 먼저 읽을 것)
- 이 스펙 실행 전에 반드시 확인: 아래 "사용할 데이터"

## 설계 변경 이력
- **2026-07-15**: internal_inconsistency 판정 로직 수정 (아래 "코드-값 일치성 검증" 참고).
  이전 로직은 used_values에 있는데 code가 안 쓴 값(unused key)이 있으면 무조건 Fail
  처리했으나, 이 때문에 실제로는 정답인 재시도 결과가 억울하게 폐기되는 사례가 확인됨
  (예: q_uid=9307945c, 정답 432를 정확히 도출했으나 unused key 하나 때문에 폐기됨).
  Llama3.2-3B에서 검증한 결과 internal_inconsistency 오탐이 16건→4건(-75%)으로
  줄었고, Condition2 EM이 56.25%→58.75%(+2.5%p) 개선됨(자세한 내용:
  bugfix_before_after_comparison.md).
- **2026-07-16**: 재시도 피드백을 옵션 A(항목명만 전달)에서 옵션 C(Verifier가 확인한
  값까지 전달)로 변경. 4개 실측 사례(diagnostic_only_qwencalc_gpt54miniverifier
  실험의 재시도 케이스) 검증 결과, 옵션 A/B(문구만 다르게 준 경우)는 재시도 성공률에
  차이가 없었던 반면, 옵션 C는 4건 중 3건을 정답으로 전환시켰다(상세:
  retry_feedback_option_c_4cases.md). 다만 Verifier 자신이 틀린 값을 제시한 경우
  (hallucination) Calculator가 이를 무비판적으로 수용해 오히려 정답을 훼손한 사례도
  1건 확인됨(q_uid=a3e21403) — 이 트레이드오프는 논문 Limitations에 명시할 것.
  순효과가 명확히 긍정적이라 옵션 C를 기본값으로 채택함.
- **재시도 지표 재정의**: 기존 `n_retry_fixed`(정의 모호, "2차 Verifier가 통과시켰는가"만
  측정)를 폐기하고 아래 4개 지표로 대체함(자세한 배경: 위 2026-07-15 관련 로그 분석).
- **2026-08-07 (멘토 미팅 8/7 피드백)**: Condition 3(Proposed)를 3a(블라인드형,
  기존)와 3b(열람형, 신규)로 분리함. 기존 블라인드형 Verifier는 Calculator가 "사용했다"고
  주장한 값만 표에서 재확인하므로, 값 추출은 정확했지만 계산 로직 자체가 틀린 경우(예:
  "총계"를 물었는데 code가 세부 항목 하나만 참조)를 원천적으로 못 잡아낸다는 지적을 받음
  — 지금까지 실험에서 Proposed가 PoT 대비 개선이 거의 없던 핵심 원인으로 지목됨. 이를
  해결하기 위해 3b는 Verifier에게 Calculator의 `code`와 질문 원문까지 함께 전달해서 값
  불일치(value_mismatch)와 로직 불일치(logic_mismatch)를 구분해 판정하게 함. 3a는 코드
  변경 없이 그대로 유지해 두 구조를 ablation 비교함(자세한 배경 및 파일럿 설계:
  verifier_redesign_literate_pilot.md).

## 사용할 데이터

### PoC 단계 (완료)
- 파일: `poc_sample.json` (filtered_dev.json에서 seed=7로 뽑은 80문항)
- 이 단계에서는 이 파일만 사용했음. Qwen2.5-3B-Instruct, Llama3.2-3B로 실행 완료.

### 본 실험 단계 (Verifier 크기 스윕, 멘토님 확인 후 진행)
- Calculator 기준선: Qwen3-4B-Instruct (기존 Qwen2.5-3B PoC와는 세대가 달라 새로
  기준선을 잡음. 스모크 테스트로 값 추출 정확도가 크게 개선됨을 확인함)
- Verifier 스윕 대상: Qwen3-4B → 14B → (필요시 8B/32B 추가)
- 평가용: filtered_dev.json + filtered_test.json (필요시 filtered_train.json 잔여분 추가)
- Few-shot 예시용: few_shot_examples.json (PoT/Proposed 조건에서 시연 예제로만 사용, 평가 대상 아님)
- 주의: few_shot_examples.json에 있는 문항은 평가 표본에 절대 포함하지 말 것
- 1단계(빠른 스윕): 150~200문항. 2단계(확정): 변곡점 근처 크기만 표본 확대 + 통계 검정.

## 데이터 구조
poc_sample.json은 원본 TAT-QA 문서 구조를 유지함:
`{table, paragraphs, questions: [...]}`
questions 배열의 각 원소를 순회하며 하나씩 처리. table은 문서 레벨에 있으므로 매 문항마다 상위 문서의 table을 참조해서 프롬프트에 넣을 것.

## 공통 입력 형식
- table: 2차원 배열 (원본 TAT-QA 형식 그대로)
- question: 질문 텍스트
- (Standard/PoT 조건에서는 Verifier 미사용)

---

## Agent 1 — Calculator

### 시스템 프롬프트
```
당신은 재무제표 표를 보고 산술 질문에 답하는 어시스턴트입니다.
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
}
```

Llama 계열 사용 시 Ollama API 호출에 `format: "json"` 파라미터를 추가로 지정할 것
(JSON 포맷 준수율이 크게 개선됨을 확인함). 아울러 few-shot 예시 1~2개를
few_shot_examples.json에서 뽑아 시스템 프롬프트 뒤에 추가할 것 — 3B급 모델은
지시문만으로는 포맷 준수가 불안정함(Llama3.2-3B PoT: JSON 강제+few-shot 적용 전
1.25% → 적용 후 57.5%).

### 유저 프롬프트 템플릿
```
표:
{table_as_text}

질문: {question}
```

### 재시도 시 추가되는 피드백 (Verifier가 불일치 발견했을 때) — 옵션 C (2026-07-16 채택)
```
아래 항목에서 값이 표와 일치하지 않는 것으로 확인되었습니다.
- {mismatched_item}: 이전에 사용한 값 {calculator_value}는 검증 결과 {verifier_found_value}로 확인되었습니다.
이 값을 참고하여 계산을 다시 수행하고, 표에서 실제로 이 값이 맞는지 스스로도 한 번 더
확인한 뒤 답하세요.
(위와 동일한 JSON 형식으로 재답변)
```
`{verifier_found_value}`는 Verifier의 `verified_values`에서 해당 항목의 값을 그대로 전달한다.

---

## Agent 2 — Verifier

### 시스템 프롬프트
```
당신은 다른 어시스턴트(Calculator)가 재무제표 표에서 값을 추출해 계산한 결과를 검증하는 검증자입니다.
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
}
```

### 유저 프롬프트 템플릿
```
표:
{table_as_text}

다음은 다른 어시스턴트가 계산에 사용했다고 주장하는 값들입니다. 이 값들이 표와 일치하는지 독립적으로 확인하세요.
{calculator_used_values_as_json}
```

### 알려진 한계 — Verifier 오탐(hallucination)
강한 모델(gpt-5.4-mini)조차 mismatch 판정 중 상당수가 오탐이었음을 확인함
(diagnostic_only 실험 기준 Precision 54.5%). 일부는 표에 없는 값을 지어내는
hallucination도 관찰됨(q_uid=a3e21403). 옵션 C 채택 이후 이 오탐이 그대로
Calculator에게 전달되어 정답을 훼손할 위험이 있다는 점을 인지할 것 — 완전한
해결책은 아직 없으며, 논문 Limitations에 명시함.

---

## Condition 1: Standard Prompting

### 시스템 프롬프트
```
당신은 재무제표 표를 보고 질문에 답하는 어시스턴트입니다.
표를 보고 질문에 대한 최종 숫자 답만 제시하세요. 풀이 과정은 출력하지 마세요.
반드시 아래 JSON 형식으로만 답하세요.

{"answer": <숫자>}
```

### 유저 프롬프트
```
표: {table_as_text}
질문: {question}
```

### 특징
- 코드 생성/실행 없음. 모델이 "암산"으로 바로 답함.
- 재시도 없음.
- Calculator/Verifier 구조 자체가 없는, 순수 zero-shot 베이스라인.

---

## Condition 2: PoT (Program-of-Thought)

### 시스템 프롬프트
(Calculator 시스템 프롬프트와 동일 — 위 "Agent 1 — Calculator" 섹션 그대로 사용)

### 유저 프롬프트
(Calculator 유저 프롬프트 템플릿과 동일)

### 특징
- Calculator 1회 호출로 끝남.
- code 필드의 파이썬 코드를 실제로 실행해서 나온 결과값을 최종 답으로 채택
  (모델이 자체 계산한 "answer" 필드가 아니라, code를 sandbox에서 실행한 결과를 써야 함 —
  이래야 진짜 PoT 방식이 됨. 이 부분 반드시 구현할 것.)
- Verifier 호출 없음. 재시도 없음.
- 아래 "코드-값 일치성 검증"은 Condition 2에도 동일하게 적용함(공정 비교를 위해).

---

## Condition 3a: Proposed Method (블라인드형 Verifier)

### 흐름
1. Calculator 호출 → used_values, code, answer 받음
2. code를 실제로 실행해서 결과값 획득 (Condition 2와 동일한 실행 로직 재사용)
3. 아래 "코드-값 일치성 검증" 통과 못 하면 즉시 Fail (Verifier 호출 안 함)
4. Verifier 호출 (Calculator의 code/answer는 전달하지 않음, used_values만 전달)
5. mismatches가 비어있으면 → 실행 결과값을 최종 answer로 채택, 종료
6. mismatches가 있으면 → 옵션 C 피드백(항목명 + Verifier 확인값)을 Calculator에
   전달, 1회 재시도
7. 재시도 후 다시 코드-값 일치성 검증 → 통과 못 하면 즉시 Fail
8. 통과했으면 다시 Verifier 검증 → 그래도 mismatch면 최종 실패(Fail)로 기록 (EM 채점 시 오답 처리)

### 특징
- Calculator 최대 2회, Verifier 최대 2회 호출 (재시도 포함).
- 로그에 각 단계 결과를 전부 남길 것 (나중에 Precision/Recall 계산 및 에러 분석에 필요):
  - 1차 Calculator 출력, 1차 Verifier 출력, (재시도 시) 2차 Calculator 출력, 2차 Verifier 출력, 최종 채택 값
- `run_poc.py --conditions 3` 으로 실행 (기존 Condition 3와 동일, 이름만 3a로 재명명해서 문서화).

---

## Condition 3b: Proposed Method (열람형 Verifier, 2026-08-07 재설계)

블라인드형(3a)과 흐름은 동일하되, Verifier가 매 호출마다 `used_values`뿐 아니라
Calculator의 `code`와 질문 원문까지 함께 받는다는 점, 그리고 값 불일치와 로직
불일치를 분리해서 판정한다는 점이 다르다.

### Verifier 시스템 프롬프트 (3a와 다른 부분만)
```
당신은 다른 어시스턴트(Calculator)가 재무제표 표에서 값을 추출하고 계산한 결과를
검증하는 검증자입니다. 아래 정보를 모두 전달받습니다: 원본 표, 질문, Calculator가
"사용했다"고 주장하는 항목명과 값의 목록, 그리고 Calculator가 실제로 실행한 파이썬 코드.

절차:
1. 표를 처음부터 독립적으로 읽고, 주어진 각 항목명에 해당하는 실제 값을 스스로 찾습니다.
   Calculator가 주장한 값과 대조하여 값 자체의 불일치가 있는지 확인합니다.
2. Calculator의 code를 검토하여, 질문이 요구하는 계산과 code가 실제로 수행하는 연산이
   일치하는지 확인합니다. 예를 들어 질문이 "총계"를 묻는데 code가 세부 항목 하나만
   참조하거나, 질문이 "감소분"을 묻는데 code가 반대 부호로 계산하는 경우 등을 확인합니다.
3. 값 불일치와 로직 불일치를 구분하여 아래 JSON 형식으로만 답하세요.

{
  "verified_values": {"<항목명>": <Verifier가 독립적으로 찾은 값>, ...},
  "value_mismatches": ["<값 자체가 표와 다른 항목명만 나열>"],
  "logic_mismatch": true,
  "logic_mismatch_reason": "<logic_mismatch가 true인 경우, 질문 의도와 code가 실제로
    다른 이유를 한 문장으로. false면 null>"
}
```

### Verifier 유저 프롬프트 템플릿
```
표:
{table_as_text}

질문: {question}

다음은 다른 어시스턴트가 계산에 사용했다고 주장하는 값과 실제로 실행한 코드입니다.
값이 표와 일치하는지, 그리고 코드가 질문이 요구하는 계산과 일치하는지 독립적으로
확인하세요.
사용한 값: {calculator_used_values_as_json}
실행한 코드:
{calculator_code}
```

### 재시도 피드백 — 옵션 D (값/로직 불일치 구분)
```
아래 문제가 발견되었습니다.
- 값 불일치: {item}, 이전에 사용한 값 {calculator_value}, 표를 재확인하세요. (해당 시,
  valid_mismatches 각 항목마다 반복)
- 계산 로직 문제: {logic_mismatch_reason} (logic_mismatch가 true인 경우만)
이를 반영하여 코드를 다시 작성하고 답하세요.
(위와 동일한 JSON 형식으로 재답변)
```

### 흐름 (3a와 다른 부분만)
4. Verifier 호출 — used_values + code + question 함께 전달 (3a는 used_values만)
5. `value_mismatches`도 없고 `logic_mismatch`도 false → 실행 결과값을 최종 answer로
   채택, 종료
6. `value_mismatches`(유효한 것 기준)가 있거나 `logic_mismatch`가 true → 옵션 D
   피드백을 Calculator에 전달, 1회 재시도
8. 재시도 후에도 유효한 value_mismatch가 있거나 logic_mismatch가 true → 최종
   실패(Fail)로 기록

### 실행/로그
- `run_poc.py --conditions 3b` 로 실행. 출력 파일은 `results/poc_{model}_condition3b.jsonl`.
- `verifier_call_1`/`verifier_call_2`에 `value_mismatches`, `valid_mismatches`,
  `invalid_mismatch_items`, `self_consistent_mismatch_items`, `logic_mismatch`,
  `logic_mismatch_reason` 필드가 추가로 남는다(3a의 `mismatches` 대신).
- `poc_summary.json`의 `condition3b`에 `n_logic_mismatch_flagged`(logic_mismatch가
  true로 판정된 재시도 건수)가 추가로 집계된다.
- 3a와 3b는 같은 `calculator_call_1`(shared calculator)을 공유하므로, 두 조건을
  같이 돌리면 EM 차이는 순수하게 "Verifier가 code까지 보는지 여부"만 반영한다.

---

## 코드-값 일치성 검증 (internal_inconsistency 판정, 2026-07-15 수정)

Calculator가 응답할 때마다(1차/2차 공통), `code`에서 리터럴로 등장하는 모든 숫자를
정규식으로 추출해 아래 기준으로 판정한다:

1. **`used_values`에 있는데 `code`에서 안 쓴 키** (unused key): 무해함. Fail 처리하지
   않는다. 로그에 `unused_keys` 필드로만 기록한다.
2. **`code`에서 쓴 숫자인데 `used_values`에도 없고 흔한 상수(0, 1, 100 등 스케일/부호
   변환용)도 아닌 경우**: 이 경우만 `internal_inconsistency: true`로 판정하고 해당
   시도를 즉시 Fail 처리한다(Verifier 호출까지 가지 않음).

이전(수정 전) 로직은 1번 케이스도 무조건 Fail 처리했으며, 이 때문에 실제로 정답인
계산이 억울하게 폐기되는 사례가 다수 확인되어 위와 같이 수정함(설계 변경 이력 참고).

---

## 실행 환경 설정
- Decoding: temperature=0, 모든 조건 동일하게 고정 (재현성 확보). 단, temperature=0이어도
  API가 완전히 결정적이지 않을 수 있음을 실측으로 확인함(동일 입력에 대해 같은 모델을
  독립적으로 두 번 호출 시 응답이 미세하게 달라지는 사례 존재) — 문항 수가 적을 때는
  이 노이즈가 지표에 1문항 단위로 영향을 줄 수 있으므로, 결과 해석 시 유의할 것.
- 표 텍스트 변환: 각 행을 " | "로 구분해서 줄바꿈으로 이어붙임 (지금까지 filtered 데이터 출력에 쓴 방식과 동일하게 통일)
- 코드 실행: exec()는 5초 타임아웃, 별도 격리된 네임스페이스에서 실행. 문법/런타임 에러
  발생 시 해당 시도는 실패(Fail)로 기록하고 재시도로 넘어가지 않음 (Condition 3에서는
  Verifier 재시도와는 별개 취급). exec 전 `textwrap.dedent()`로 공통 들여쓰기를
  제거하고, 그래도 IndentationError가 나면 들여쓰기를 4칸 단위로 정규화해서 1회만
  재시도(코드 실행 레벨 전처리이며 Verifier 재시도와는 무관).
- EM 채점: TAT-QA 공식 evaluation script(원 논문 GitHub: NExTplusplus/TAT-QA, tatqa_metric.py 등) 로직을 그대로 사용. scale 필드까지 포함해서 정답과 비교할 것.

---

## 출력 결과 저장 형식

### 파일 구조
results/ 폴더 아래에 조건별 × 모델별로 나눠 저장:
```
results/
  poc_{model}_condition1.jsonl
  poc_{model}_condition2.jsonl
  poc_{model}_condition3.jsonl
  poc_{model}_condition3b.jsonl
  poc_summary.json
```
model에는 실제 모델명을 슬래시 없이 사용(예: qwen3-4b, qwen3-14b, llama3.2-3b).
진단/ablation 목적의 1회성 실행은 파일명 앞에 `diagnostic_only_` 접두어를 붙여
본 실험 결과와 섞이지 않게 한다.

### 문항 단위 로그 (jsonl, 한 줄에 시도 하나)
원본 question 객체는 필드를 재구성하지 말고 `gold`로 통째로 중첩. `table`도 매 줄에 전체 포함(에러 분석 시 원본 파일과 다시 join 안 해도 되도록):

```json
{
  "gold": {
    "uid": "...", "order": 0, "question": "...", "answer": "...",
    "derivation": "...", "answer_type": "arithmetic", "answer_from": "table",
    "rel_paragraphs": [], "req_comparison": false, "scale": "..."
  },
  "table_uid": "...",
  "table": [["..."]],

  "condition": "1 | 2 | 3 | 3b",
  "model": "qwen3:4b-instruct | qwen3:14b-instruct | llama3.2:3b | ...",

  "predicted_answer": "...",
  "is_correct": true,

  "calculator_call_1": {
    "used_values": {},
    "code": "...",
    "raw_response": "...",
    "code_exec_result": "...",
    "code_exec_error": null,
    "unused_keys": [],
    "internal_inconsistency": false
  },
  "verifier_call_1": {
    "verified_values": {},
    "mismatches": [],
    "raw_response": "..."
  },
  "retried": false,
  "calculator_call_2": null,
  "verifier_call_2": null,

  "latency_seconds": 0.0,
  "timestamp": "ISO 8601"
}
```
(Condition 1/2는 calculator_call_1만 있고 verifier_call_1은 null, retried는 항상 false)

Condition 3b의 `verifier_call_1`/`verifier_call_2`는 위 예시의 `mismatches` 대신
`value_mismatches`, `valid_mismatches`, `invalid_mismatch_items`,
`self_consistent_mismatch_items`, `logic_mismatch`(bool), `logic_mismatch_reason`
(string 또는 null) 필드를 가진다 (위 "Condition 3b" 섹션 참고).

### 요약 파일 (poc_summary.json)
```json
{
  "run_date": "...",
  "n_questions": 80,
  "results": {
    "{model_name}": {
      "condition1": {"em": 0.0, "n_correct": 0, "n_total": 0},
      "condition2": {
        "em": 0.0, "n_correct": 0, "n_total": 0,
        "n_internal_inconsistency": 0
      },
      "condition3": {
        "em": 0.0, "n_correct": 0, "n_total": 0,
        "n_internal_inconsistency": 0,
        "verifier_precision": 0.0,
        "verifier_recall": 0.0,
        "n_ambiguous": 0,
        "automated_not_manually_validated": true,
        "n_retried": 0,
        "n_retry_truly_fixed": 0,
        "n_retry_regressed": 0,
        "n_retry_verifier_satisfied_but_still_wrong": 0,
        "n_retry_invalid_mismatch": 0,
        "avg_latency_seconds": 0.0
      },
      "condition3b": {
        "...": "condition3와 동일한 필드 전부 + 아래 추가 필드",
        "n_logic_mismatch_flagged": 0
      }
    }
  }
}
```

재시도 지표 정의 (n_retry_fixed는 폐기, 아래로 대체):
- `n_retry_truly_fixed`: 1차는 gold와 불일치했으나 최종 predicted_answer는 gold와 일치
- `n_retry_regressed`: 1차는 gold와 일치했으나 최종 predicted_answer는 불일치
- `n_retry_verifier_satisfied_but_still_wrong`: 2차 Verifier가 통과시켰지만 최종
  predicted_answer는 여전히 gold와 불일치
- `n_retry_invalid_mismatch`: mismatch 항목명이 실제 used_values 키와 매칭되지 않아
  무효 처리되어 재시도가 스킵된 경우

### 저장 시점
- 문항 하나 처리할 때마다 즉시 jsonl에 append (전체 다 돌고 한 번에 저장하지 말 것 — 중간에 에러 나서 멈춰도 여태까지 결과는 남게)
- 전체 완료 후 poc_summary.json 생성
