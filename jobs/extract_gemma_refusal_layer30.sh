#!/bin/bash
set -euo pipefail

cd /scratch/mechhack/starter_code

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/scratch/mechhack/starter_code:${PYTHONPATH:-}

/opt/conda/bin/python extract_residuals.py \
  --model_key gemma4_31b \
  --model_path /data/Gemma-4-31B-it \
  --samples_file /scratch/mechhack/datasets/refusal_probes/gemma4_31b/attacks_full.jsonl \
  --out_dir /scratch/hackathon_extracts/gemma_refusal_layer30 \
  --layers "30"
