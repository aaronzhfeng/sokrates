#!/bin/bash
# ============================================================
# ABLATION 1: w/o optionization (Raw CoT)
# ============================================================
# 
# This ablation removes the optionized format (<Option> tags) and
# uses natural language chain-of-thought reasoning instead.
#
# Pipeline:
#   1. Convert training data to raw CoT format
#   2. Train SFT on raw CoT
#   3. Generate traces (raw CoT format)
#   4. Train DPO (1 iteration, answer-only since can't verify steps)
#   5. Evaluate
#
# Uses GPUs 2-7 (avoids GPU 0)
# ============================================================

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "ABLATION 1: w/o optionization (Raw CoT)"
echo "============================================================"

GPUS="2 3 4 5 6 7"
NUM_GPUS=6
OUTPUT_BASE="outputs/ablation/raw_cot"
mkdir -p "$OUTPUT_BASE"

# ============================================================
# Step 1: Prepare Raw CoT training data
# ============================================================
echo ""
echo "[Step 1] Converting training data to raw CoT format..."
python scripts/ablation_prepare_data.py \
    --input data/processed/prontoqa_train.jsonl \
    --output "$OUTPUT_BASE/train_raw_cot.jsonl" \
    --format raw_cot

echo "✓ Created raw CoT training data"

# ============================================================
# Step 2: Train SFT on Raw CoT
# ============================================================
echo ""
echo "[Step 2] Training SFT on raw CoT format..."
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    scripts/train_sft.py \
    --data "$OUTPUT_BASE/train_raw_cot.jsonl" \
    --output-dir "$OUTPUT_BASE/sft" \
    --epochs 3 \
    --batch-size 4

echo "✓ SFT training complete"

# ============================================================
# Step 3: Merge SFT model for inference
# ============================================================
echo ""
echo "[Step 3] Merging SFT model..."
# Find the latest SFT model (uses timestamped directories)
SFT_MODEL="$OUTPUT_BASE/sft/latest/final"
if [ ! -d "$SFT_MODEL" ]; then
    # Fallback: find the most recent timestamped directory
    LATEST_DIR=$(ls -td "$OUTPUT_BASE/sft"/20* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        SFT_MODEL="$LATEST_DIR/final"
    fi
fi

if [ -f "${SFT_MODEL}/adapter_config.json" ]; then
    echo "  Merging LoRA adapter..."
    python scripts/merge_lora_adapter.py \
        --adapter "$SFT_MODEL" \
        --output "$OUTPUT_BASE/sft/merged"
    SFT_MODEL="$OUTPUT_BASE/sft/merged"
fi
echo "✓ Using model: $SFT_MODEL"

# ============================================================
# Step 4: Generate traces (Raw CoT format)
# ============================================================
echo ""
echo "[Step 4] Generating traces with raw CoT model (6 GPUs)..."

# For raw CoT, we use a simpler prompt since there's no Option format
TRACE_OUTPUT="$OUTPUT_BASE/traces"
mkdir -p "$TRACE_OUTPUT"

# Use data parallelism across 6 GPUs
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
            --max-steps 15 \
            --skip-verify \
            --raw-cot &  # Use raw CoT format (no Option tags)
        PIDS+=($!)
    fi
done

# Wait for all to complete
for PID in "${PIDS[@]}"; do
    wait "$PID" || { echo "Error in GPU process"; exit 1; }
done

# Merge traces
find "$TRACE_OUTPUT" -name "traces.jsonl" -print0 | xargs -0 cat > "$TRACE_OUTPUT/traces.jsonl"
echo "✓ Generated $(wc -l < "$TRACE_OUTPUT/traces.jsonl") traces"

# ============================================================
# Step 5: Train DPO (answer-only since can't verify steps)
# ============================================================
echo ""
echo "[Step 5] Training DPO with answer-only preferences..."
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes=$NUM_GPUS --mixed_precision=bf16 \
    scripts/train_dpo_answer_only.py \
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
            --temperature 0.0 \
            --skip-verify \
            --raw-cot &
        PIDS+=($!)
    fi
done

for PID in "${PIDS[@]}"; do
    wait "$PID" || { echo "Error in GPU process"; exit 1; }
done

# Aggregate results
python3 << 'EOF'
import json
import glob
import os

output_dir = os.environ.get("EVAL_OUTPUT", "outputs/ablation/raw_cot/eval")
summaries = []

for f in glob.glob(os.path.join(output_dir, "gpu*/summary.json")):
    with open(f) as fp:
        summaries.append(json.load(fp))

if summaries:
    total = sum(s.get("total_traces", 0) for s in summaries)
    correct = sum(s.get("correct_traces", 0) for s in summaries)
    acc = correct / total if total > 0 else 0
    
    combined = {
        "ablation": "raw_cot",
        "total": total,
        "correct": correct,
        "accuracy": acc,
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w") as fp:
        json.dump(combined, fp, indent=2)
    
    print(f"Accuracy: {acc*100:.1f}% ({correct}/{total})")
EOF

echo ""
echo "============================================================"
echo "ABLATION 1 COMPLETE: w/o optionization (Raw CoT)"
echo "============================================================"
echo "Results saved to: $OUTPUT_BASE"
cat "$EVAL_OUTPUT/summary.json" 2>/dev/null || echo "(no summary)"

