# Mech-Interp Hackathon Submission

## Method

We use persona/trait vectors as an interpretable feature basis for prompt-side model internals.

For each prompt:
1. Run Gemma 4-31B-it on the attack prompt only.
2. Extract the final prompt-token residual at layer 30.
3. Project the activation onto 240 learned trait/persona vectors plus an assistant axis.
4. Train logistic regression probes for refusal and cyber-risk classification.

## Level 1 Results

See `results/task1/summary.json`.

Completed tasks:
- Refusal-Gemma
- Cyber Probe-1: dual_use vs benign
- Cyber Probe-2: high_risk_dual_use vs dual_use ∪ benign
- Cyber Probe-3: prohibited vs rest

Qwen was not completed due to time.

## Level 2

We use prompt-specific trait contributions:
`contribution_i = standardized_projection_i * classifier_weight_i`

Then we identify:
- missing compliance-associated traits
- overactive refusal-associated traits

We translate trait names back into the original trait instruction text and use those as natural-language rewording guidance.

Behavior verification reruns Gemma locally and uses an OpenAI judge to classify refusal vs compliance.

## Important Files

### Code
- `code/gemma_layer30_traitproj_all_tasks.py`: extract layer-30 activations, compute trait projections, train probes.
- `code/disrupt4_updated.py`: prompt-specific trait-guided rewrite loop.
- `code/behavior_check_edits.py`: rerun Gemma on edited prompts and judge behavior.
- `code/4_vectors_traits40_hf.py`: compute trait vectors.
- `code/5_compute_assistant_axis_selected_layers.py`: compute assistant axis.
- `code/extract_residuals.py`: original residual extractor scaffold.

### Results
- `results/task1/summary.json`: Task 1 metrics.
- `results/task1/top_trait_coefficients.json`: classifier trait coefficients.
- `results/task2/*.summary.json`: Level 2 proxy/behavior summaries.

### Jobs
- `jobs/*.sh`: RunAI job scripts used for extraction, editing, and behavior verification.

## Notes

Large activation/extract `.pt` files are not included because they are too large. They can be regenerated from the scripts using the fixed train/test splits in the hackathon datasets.
