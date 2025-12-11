#!/bin/bash
# ============================================================
# ABLATION 2: w/o Thought (Action-only)
# ============================================================
# 
# This ablation removes the Thought lines and keeps only Action lines.
# The model still uses the optionized format, but without explanations.
#
# Pipeline:
#   1. Convert training data to action-only format
#   2. Train SFT on action-only
#   3. Generate traces
#   4. Train DPO (1 iteration, with solver verification)
#   5. Evaluate
#
# Uses GPUs 2-7 (avoids GPU 0)
# ============================================================

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "ABLATION 2: w/o Thought (Action-only)"
echo "============================================================"

GPUS="2 3 4 5 6 7"
NUM_GPUS=6
OUTPUT_BASE="outputs/ablation/action_only"
mkdir -p "$OUTPUT_BASE"

# ============================================================
# Step 1: Prepare Action-only training data
# ============================================================
echo ""
echo "[Step 1] Converting training data to action-only format..."
python scripts/ablation_prepare_data.py \
    --input data/processed/prontoqa_train.jsonl \
    --output "$OUTPUT_BASE/train_action_only.jsonl" \
    --format action_only

echo "✓ Created action-only training data"

# ============================================================
# Step 2: Train SFT on Action-only
# ============================================================
echo ""
echo "[Step 2] Training SFT on action-only format..."
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    scripts/train_sft.py \
    --data "$OUTPUT_BASE/train_action_only.jsonl" \
    --output-dir "$OUTPUT_BASE/sft" \
    --epochs 3 \
    --batch-size 4

echo "✓ SFT training complete"

# ============================================================
# Step 3: Merge SFT model for inference
# ============================================================
echo ""
echo "[Step 3] Merging SFT model..."
SFT_MODEL="$OUTPUT_BASE/sft/final"
if [ -f "${SFT_MODEL}/adapter_config.json" ]; then
    python scripts/merge_lora_adapter.py \
        --adapter "$SFT_MODEL" \
        --output "$OUTPUT_BASE/sft/merged"
    SFT_MODEL="$OUTPUT_BASE/sft/merged"
fi
echo "✓ Using model: $SFT_MODEL"

# ============================================================
# Step 4: Generate traces
# ============================================================
echo ""
echo "[Step 4] Generating traces with action-only model (6 GPUs)..."

TRACE_OUTPUT="$OUTPUT_BASE/traces"
mkdir -p "$TRACE_OUTPUT"

TOTAL_PROBLEMS=1594
PPG=$((TOTAL_PROBLEMS / NUM_GPUS + 1))

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$(echo $GPUS | cut -d' ' -f $((i + 1)))
    START_IDX=$((i * PPG))
    
    if [ "$START_IDX" -lt "$TOTAL_PROBLEMS" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/generate_traces_vllm.py \
            --model "$SFT_MODEL" \
            --data data/processed/prontoqa_train.jsonl \
            --output "$TRACE_OUTPUT/gpu${GPU_ID}" \
            --num-problems "$PPG" \
            --start-idx "$START_IDX" \
            --samples-per-problem 2 \
            --temperature 0.5 \
            --max-steps 15 &
            # Note: solver verification still works since Options are preserved
        PIDS+=($!)
    fi
done

for PID in "${PIDS[@]}"; do
    wait "$PID" || { echo "Error in GPU process"; exit 1; }
done

# Merge traces
find "$TRACE_OUTPUT" -name "traces.jsonl" -print0 | xargs -0 cat > "$TRACE_OUTPUT/traces.jsonl"
echo "✓ Generated $(wc -l < "$TRACE_OUTPUT/traces.jsonl") traces"

# ============================================================
# Step 5: Train DPO (with solver verification)
# ============================================================
echo ""
echo "[Step 5] Training DPO with step validity preferences..."
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces "$TRACE_OUTPUT/traces.jsonl" \
    --model "$SFT_MODEL" \
    --output "$OUTPUT_BASE/dpo" \
    --num-epochs 1 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-6 \
    --beta 0.1

echo "✓ DPO training complete"

# ============================================================
# Step 6: Merge DPO model and Evaluate
# ============================================================
echo ""
echo "[Step 6] Evaluating..."
DPO_MODEL="$OUTPUT_BASE/dpo/final"
if [ -f "${DPO_MODEL}/adapter_config.json" ]; then
    python scripts/merge_lora_adapter.py \
        --adapter "$DPO_MODEL" \
        --output "$OUTPUT_BASE/dpo/merged"
    DPO_MODEL="$OUTPUT_BASE/dpo/merged"
fi

EVAL_OUTPUT="$OUTPUT_BASE/eval"
mkdir -p "$EVAL_OUTPUT"

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$(echo $GPUS | cut -d' ' -f $((i + 1)))
    START_IDX=$((i * PPG))
    
    if [ "$START_IDX" -lt "$TOTAL_PROBLEMS" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/generate_traces_vllm.py \
            --model "$DPO_MODEL" \
            --data data/processed/prontoqa_test.jsonl \
            --output "$EVAL_OUTPUT/gpu${GPU_ID}" \
            --num-problems "$PPG" \
            --start-idx "$START_IDX" \
            --samples-per-problem 1 \
            --temperature 0.0 &
        PIDS+=($!)
    fi
done

for PID in "${PIDS[@]}"; do
    wait "$PID" || { echo "Error in GPU process"; exit 1; }
done

# Aggregate results
export EVAL_OUTPUT
python3 << 'EOF'
import json
import glob
import os

output_dir = os.environ.get("EVAL_OUTPUT", "outputs/ablation/action_only/eval")
summaries = []

for f in glob.glob(os.path.join(output_dir, "gpu*/summary.json")):
    with open(f) as fp:
        summaries.append(json.load(fp))

if summaries:
    total = sum(s.get("total_traces", 0) for s in summaries)
    correct = sum(s.get("correct_traces", 0) for s in summaries)
    valid_steps = sum(s.get("valid_steps", 0) for s in summaries)
    total_steps = sum(s.get("total_steps", 0) for s in summaries)
    
    acc = correct / total if total > 0 else 0
    step_val = valid_steps / total_steps if total_steps > 0 else 0
    
    combined = {
        "ablation": "action_only",
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "valid_steps": valid_steps,
        "total_steps": total_steps,
        "step_validity": step_val,
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w") as fp:
        json.dump(combined, fp, indent=2)
    
    print(f"Accuracy: {acc*100:.1f}% ({correct}/{total})")
    print(f"Step Validity: {step_val*100:.1f}%")
EOF

echo ""
echo "============================================================"
echo "ABLATION 2 COMPLETE: w/o Thought (Action-only)"
echo "============================================================"
echo "Results saved to: $OUTPUT_BASE"
cat "$EVAL_OUTPUT/summary.json" 2>/dev/null || echo "(no summary)"

