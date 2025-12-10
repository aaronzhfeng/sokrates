#!/bin/bash
# SOKRATES OaK Loop - Iterative Training
# 
# Runs the full OaK loop with two-stage pipeline:
#   For each iteration:
#     1. Generate traces with current model
#     2. Train DPO on those traces
#     3. Use DPO model for next iteration
#
# Usage:
#   ./scripts/run_oak_loop.sh [NUM_ITERATIONS] [NUM_PROBLEMS] [GPU_IDS]
#
# Example:
#   ./scripts/run_oak_loop.sh 2 1500 "0,1"

set -e

# Configuration
NUM_ITERATIONS=${1:-2}
NUM_PROBLEMS=${2:-1500}
GPU_IDS=${3:-"0"}
SAMPLES_PER_PROBLEM=${4:-2}

SFT_MODEL="outputs/sft/20251209_150417/final"
TRAIN_DATA="data/processed/prontoqa_train.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="outputs/oak_loop_$TIMESTAMP"

echo "============================================================"
echo "SOKRATES OaK Loop"
echo "============================================================"
echo "Iterations: $NUM_ITERATIONS"
echo "Problems: $NUM_PROBLEMS"
echo "Samples per problem: $SAMPLES_PER_PROBLEM"
echo "GPUs: $GPU_IDS"
echo "Output: $OUTPUT_BASE"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_BASE"

# Save config
cat > "$OUTPUT_BASE/config.json" << EOF
{
    "num_iterations": $NUM_ITERATIONS,
    "num_problems": $NUM_PROBLEMS,
    "samples_per_problem": $SAMPLES_PER_PROBLEM,
    "gpu_ids": "$GPU_IDS",
    "sft_model": "$SFT_MODEL",
    "train_data": "$TRAIN_DATA",
    "timestamp": "$TIMESTAMP"
}
EOF

CURRENT_MODEL="$SFT_MODEL"

for ((i=0; i<NUM_ITERATIONS; i++)); do
    echo ""
    echo "============================================================"
    echo "ITERATION $((i+1))/$NUM_ITERATIONS"
    echo "============================================================"
    echo "Model: $CURRENT_MODEL"
    
    TRACE_DIR="$OUTPUT_BASE/traces_iter$i"
    DPO_DIR="$OUTPUT_BASE/dpo_iter$i"
    
    # Stage 1: Generate traces
    echo ""
    echo "[Stage 1] Generating traces..."
    CUDA_VISIBLE_DEVICES=$GPU_IDS python scripts/generate_traces.py \
        --model "$CURRENT_MODEL" \
        --data "$TRAIN_DATA" \
        --output "$TRACE_DIR" \
        --num-problems $NUM_PROBLEMS \
        --samples-per-problem $SAMPLES_PER_PROBLEM \
        --max-steps 6 \
        --temperature 0.0
    
    # Check trace generation succeeded
    if [ ! -f "$TRACE_DIR/traces.jsonl" ]; then
        echo "ERROR: Trace generation failed"
        exit 1
    fi
    
    # Print trace summary
    echo ""
    echo "Trace Summary:"
    cat "$TRACE_DIR/summary.json" | python -m json.tool
    
    # Stage 2: Train DPO
    echo ""
    echo "[Stage 2] Training DPO..."
    
    # Count GPUs for accelerate
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    
    if [ $NUM_GPUS -gt 1 ]; then
        CUDA_VISIBLE_DEVICES=$GPU_IDS accelerate launch \
            --num_processes=$NUM_GPUS \
            --mixed_precision=bf16 \
            scripts/train_dpo_from_traces.py \
            --traces "$TRACE_DIR/traces.jsonl" \
            --model "$CURRENT_MODEL" \
            --output "$DPO_DIR" \
            --num-epochs 1 \
            --batch-size 2 \
            --gradient-accumulation-steps 4 \
            --beta 0.1
    else
        CUDA_VISIBLE_DEVICES=$GPU_IDS python scripts/train_dpo_from_traces.py \
            --traces "$TRACE_DIR/traces.jsonl" \
            --model "$CURRENT_MODEL" \
            --output "$DPO_DIR" \
            --num-epochs 1 \
            --batch-size 2 \
            --gradient-accumulation-steps 4 \
            --beta 0.1
    fi
    
    # Check DPO succeeded
    if [ ! -d "$DPO_DIR/final" ]; then
        echo "ERROR: DPO training failed"
        exit 1
    fi
    
    # Update model for next iteration
    CURRENT_MODEL="$DPO_DIR/final"
    
    echo ""
    echo "Iteration $((i+1)) complete!"
    echo "New model: $CURRENT_MODEL"
done

echo ""
echo "============================================================"
echo "OaK Loop Complete!"
echo "============================================================"
echo "Final model: $CURRENT_MODEL"
echo "All outputs: $OUTPUT_BASE"
echo ""
echo "Directory structure:"
ls -la "$OUTPUT_BASE"
echo "============================================================"

# Create symlink to latest
ln -sfn "$OUTPUT_BASE" outputs/oak_loop_latest

