#!/bin/bash
# ============================================================
# ABLATION 3: w/o solver verification (Answer-only DPO)
# ============================================================
# 
# This ablation uses the same SFT model but trains DPO with
# answer-only preferences (ignoring step validity).
#
# This simulates what happens when we don't have a solver to
# verify logical validity of reasoning steps.
#
# Pipeline:
#   1. Use existing SFT model (no retraining needed)
#   2. Generate traces
#   3. Train DPO with answer-only preferences (1 iteration)
#   4. Evaluate
#
# Uses GPUs 2-7 (avoids GPU 0)
# ============================================================

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "ABLATION 3: w/o solver verification (Answer-only DPO)"
echo "============================================================"

GPUS="6 7"
NUM_GPUS=2
OUTPUT_BASE="outputs/ablation/answer_only_dpo"
mkdir -p "$OUTPUT_BASE"

# ============================================================
# Step 1: Use existing SFT model
# ============================================================
echo ""
echo "[Step 1] Using existing SFT model..."
SFT_MODEL="outputs/sft/latest/merged"
if [ ! -d "$SFT_MODEL" ]; then
    SFT_MODEL="outputs/sft/latest/final"
    if [ -f "${SFT_MODEL}/adapter_config.json" ]; then
        echo "  Merging SFT adapter..."
        python scripts/merge_lora_adapter.py \
            --adapter "$SFT_MODEL" \
            --output "outputs/sft/latest/merged"
        SFT_MODEL="outputs/sft/latest/merged"
    fi
fi
echo "✓ Using SFT model: $SFT_MODEL"

# ============================================================
# Step 2: Generate traces (if not already available)
# ============================================================
echo ""
echo "[Step 2] Generating traces..."

TRACE_OUTPUT="$OUTPUT_BASE/traces"
mkdir -p "$TRACE_OUTPUT"

# Check if we can reuse existing traces
EXISTING_TRACES="outputs/traces/iter0/traces.jsonl"
if [ -f "$EXISTING_TRACES" ]; then
    echo "  Found existing traces at $EXISTING_TRACES"
    echo "  Copying..."
    cp "$EXISTING_TRACES" "$TRACE_OUTPUT/traces.jsonl"
else
    echo "  Generating new traces (6 GPUs)..."
    
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
            PIDS+=($!)
        fi
    done
    
    for PID in "${PIDS[@]}"; do
        wait "$PID" || { echo "Error in GPU process"; exit 1; }
    done
    
    # Merge traces
    find "$TRACE_OUTPUT" -name "traces.jsonl" -print0 | xargs -0 cat > "$TRACE_OUTPUT/traces.jsonl"
fi

echo "✓ Traces: $(wc -l < "$TRACE_OUTPUT/traces.jsonl")"

# ============================================================
# Step 3: Train DPO with ANSWER-ONLY preferences
# ============================================================
echo ""
echo "[Step 3] Training DPO with answer-only preferences..."
echo "  (This ignores step validity - simulates 'no solver')"

CUDA_VISIBLE_DEVICES=7 python scripts/train_dpo_answer_only.py \
    --traces "$TRACE_OUTPUT/traces.jsonl" \
    --model "$SFT_MODEL" \
    --output "$OUTPUT_BASE/dpo" \
    --num-epochs 1 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-6 \
    --beta 0.1

echo "✓ DPO training complete (answer-only)"

# ============================================================
# Step 4: Merge DPO model and Evaluate
# ============================================================
echo ""
echo "[Step 4] Evaluating..."
DPO_MODEL="$OUTPUT_BASE/dpo/final"
if [ -f "${DPO_MODEL}/adapter_config.json" ]; then
    python scripts/merge_lora_adapter.py \
        --adapter "$DPO_MODEL" \
        --output "$OUTPUT_BASE/dpo/merged"
    DPO_MODEL="$OUTPUT_BASE/dpo/merged"
fi

EVAL_OUTPUT="$OUTPUT_BASE/eval"
mkdir -p "$EVAL_OUTPUT"

TOTAL_PROBLEMS=1594
PPG=$((TOTAL_PROBLEMS / NUM_GPUS + 1))

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

output_dir = os.environ.get("EVAL_OUTPUT", "outputs/ablation/answer_only_dpo/eval")
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
        "ablation": "answer_only_dpo",
        "description": "DPO trained without solver (answer-only preferences)",
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
echo "ABLATION 3 COMPLETE: w/o solver (Answer-only DPO)"
echo "============================================================"
echo "Results saved to: $OUTPUT_BASE"
cat "$EVAL_OUTPUT/summary.json" 2>/dev/null || echo "(no summary)"

