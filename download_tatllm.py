#!/usr/bin/env python3
"""Download next-tat/tat-llm-7b-fft (TAT-LLM 7B, SOTA-comparison baseline) from
Hugging Face into a local directory, for use by run_tatllm_eval.py.

TAT-LLM-7B-FFT is a full fine-tune of a 7B base model on the FinQA/TAT-QA/TAT-DQA
SFT data (see https://github.com/fengbinzhu/TAT-LLM). It is run once, single-call,
no Calculator/Verifier/retry structure -- see run_tatllm_eval.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
DEFAULT_REPO_ID = "next-tat/tat-llm-7b-fft"
DEFAULT_OUT_DIR = ROOT / "models" / "tat-llm-7b-fft"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                         help="local directory to download the model into")
    parser.add_argument("--revision", default=None, help="specific revision/commit/tag (default: main)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} -> {out_dir} ...")
    local_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=out_dir,
    )
    print(f"Done. Model files at: {local_path}")
    print(f"Run with: python run_tatllm_eval.py --model-dir {out_dir}")


if __name__ == "__main__":
    main()
