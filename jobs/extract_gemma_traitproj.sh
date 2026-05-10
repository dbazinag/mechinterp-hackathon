#!/bin/bash
set -euo pipefail

cd /scratch/mechhack/starter_code

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/scratch/mechhack/starter_code:${PYTHONPATH:-}

/opt/conda/bin/python extract_residuals.py \
  --model_key gemma4_31b \
  --model_path /data/Gemma-4-31B-it \
  --samples_file /scratch/hackathon_classifier_data/gemma_traitproj_all.jsonl \
  --out_dir /scratch/hackathon_extracts/gemma_traitproj_layers_15_20_25_30_35_40_45 \
  --layers "15,20,25,30,35,40,45"
