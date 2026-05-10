#!/bin/bash
set -euo pipefail

cd /scratch/assistant-axis-llama3.1-8B

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/opt/conda/bin/python full_trait_pipeline_hackathon/gemma_layer30_traitproj_all_tasks.py \
  --mode extract \
  --repo_root /scratch/mechhack \
  --model_path /data/Gemma-4-31B-it \
  --out_root /scratch/hackathon_traitproj_layer30_all \
  --trait_vector_dir /scratch/gemma_trait_output/traits40_vectors/pre_generation_last_token/all_traits_no_filter \
  --assistant_axis_path /scratch/gemma_trait_output/traits40_axes/pre_generation_last_token/all_traits_no_filter/assistant_axis_pc1.pt \
  --layer_idx 30 \
  --num_shards 8 \
  --shard_idx 0
