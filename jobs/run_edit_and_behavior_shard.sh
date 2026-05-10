#!/bin/bash
set -euo pipefail

SHARD_ID="${SHARD_ID:?Need SHARD_ID}"

cd /scratch/assistant-axis-llama3.1-8B/full_trait_pipeline_hackathon

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export OPENAI_EDITOR_MODEL="gpt-4.1-mini"
export OPENAI_JUDGE_MODEL="gpt-4.1-mini"

EVAL="/scratch/trait_disrupt_shards/eval/attribution_eval_shard${SHARD_ID}.jsonl"
EDIT_OUT="/scratch/trait_disrupt_shards/results/edits_shard${SHARD_ID}.jsonl"
BEHAVIOR_OUT="/scratch/trait_disrupt_shards/results/behavior_shard${SHARD_ID}.jsonl"

echo "=== SHARD ${SHARD_ID} ==="
echo "Eval: $EVAL"
echo "Edit out: $EDIT_OUT"
echo "Behavior out: $BEHAVIOR_OUT"

echo "=== Running trait-guided editing ==="
/opt/conda/bin/python disrupt4_updated.py \
  --mode local_test \
  --limit 0 \
  --max_total_edits 1000 \
  --eval_set "$EVAL" \
  --probe_path /scratch/hackathon_traitproj_layer30_all/gemma_layer30_trait_projection_probe.pt \
  --trait_vector_dir /scratch/gemma_trait_output/traits40_vectors/pre_generation_last_token/all_traits_no_filter \
  --assistant_axis_path /scratch/gemma_trait_output/traits40_axes/pre_generation_last_token/all_traits_no_filter/assistant_axis_pc1.pt \
  --model_path /data/Gemma-4-31B-it \
  --layer_idx 30 \
  --output "$EDIT_OUT"

echo "=== Running Gemma behavior check + OpenAI judge ==="
/opt/conda/bin/python behavior_check_edits.py \
  --input_jsonl "$EDIT_OUT" \
  --model_path /data/Gemma-4-31B-it \
  --output_jsonl "$BEHAVIOR_OUT" \
  --max_new_tokens 256 \
  --limit 0 \
  --no_save_responses

echo "=== Done shard ${SHARD_ID} ==="
cat "${EDIT_OUT%.jsonl}.summary.json" || true
cat "${BEHAVIOR_OUT%.jsonl}.summary.json" || true
