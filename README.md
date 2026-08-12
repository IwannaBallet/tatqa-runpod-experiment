# tatqa-runpod-experiment

이 저장소는 Verifier 재설계(열람형, Condition 3b) 실험과 1.7B/4B/8B
재실행, TAT-LLM 7B(SOTA 비교용) 실행을 위한 것이며, RunPod에서 Chris가
직접 실행하는 용도입니다.

## 포함된 내용
- `run_poc.py` — Condition 3b(열람형 Verifier) 로직 포함
- `PROMPT_SPEC.md` — 열람형 Verifier 프롬프트 스펙
- `pilot_100_uids.json` / `pilot_100_uids_meta.json` — 파일럿 100문항 표본
  (재시도 이력 문항 우선 포함 + seed=31 무작위 채움)
- `poc_sample_v3.json` — 1,000문항 샘플
- `TAT-QA/` — `tatqa_metric.py`, `tatqa_utils.py` 등 평가 유틸리티
- `filtered_tatqa/` — few-shot 예시
- `requirements.txt` — 필요 Python 패키지
- `download_tatllm.py` — TAT-LLM 7B(next-tat/tat-llm-7b-fft, SOTA 비교용) 모델을
  Hugging Face에서 다운로드
- `run_tatllm_eval.py` — TAT-LLM 7B를 poc_sample_v3.json(1,000문항)에 단일 호출로
  실행하고(재시도/Calculator/Verifier 구조 없음) tatqa_metric.py로 EM/F1 채점

## 실행 예시
```bash
python run_poc.py --model qwen3:1.7b --condition 3b --data pilot_100_uids.json
python run_poc.py --model qwen3:4b --condition 3b --data pilot_100_uids.json
python run_poc.py --model qwen3:8b --condition 3b --data pilot_100_uids.json
```
(정확한 CLI 인자명은 `run_poc.py`의 argparse 정의를 확인 후 사용할 것)

## TAT-LLM 7B (SOTA 비교) 실행 예시
```bash
python download_tatllm.py                      # models/tat-llm-7b-fft 로 다운로드
python run_tatllm_eval.py --limit 5             # 스모크 테스트
python run_tatllm_eval.py                       # 전체 1,000문항 실행
```
결과: `results/tatllm_eval.jsonl`(문항별 원문 출력/예측/정오답),
`results/tatllm_summary.json`(EM/F1 집계).
