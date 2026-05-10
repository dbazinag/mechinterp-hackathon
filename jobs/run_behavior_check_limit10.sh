#!/bin/bash
set -euo pipefail

cd /scratch/assistant-axis-llama3.1-8B/full_trait_pipeline_hackathon

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export OPENAI_JUDGE_MODEL="gpt-4.1-mini"

/opt/conda/bin/python behavior_check_edits.py \
  --input_jsonl /scratch/trait_disrupt_local_test/results_limit10_v4_updated.jsonl \
  --model_path /data/Gemma-4-31B-it \
  --output_jsonl /scratch/trait_disrupt_local_test/behavior_limit10_v4_updated.jsonl \
  --max_new_tokens 256 \
  --limit 10
