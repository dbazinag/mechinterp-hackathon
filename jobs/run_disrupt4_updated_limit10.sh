#!/bin/bash
set -euo pipefail

cd /scratch/assistant-axis-llama3.1-8B/full_trait_pipeline_hackathon

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export OPENAI_EDITOR_MODEL="gpt-4.1-mini"

/opt/conda/bin/python disrupt4_updated.py \
  --mode local_test \
  --limit 10 \
  --max_total_edits 1000 \
  --eval_set /scratch/mechhack/datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
  --probe_path /scratch/hackathon_traitproj_layer30_all/gemma_layer30_trait_projection_probe.pt \
  --trait_vector_dir /scratch/gemma_trait_output/traits40_vectors/pre_generation_last_token/all_traits_no_filter \
  --assistant_axis_path /scratch/gemma_trait_output/traits40_axes/pre_generation_last_token/all_traits_no_filter/assistant_axis_pc1.pt \
  --model_path /data/Gemma-4-31B-it \
  --layer_idx 30 \
  --output /scratch/trait_disrupt_local_test/results_limit10_v4_updated.jsonl
