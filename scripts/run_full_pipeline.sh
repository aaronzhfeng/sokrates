#!/bin/bash
# SOKRATES Full Training Pipeline
# Runs SFT → OaK-DPO sequentially
# Usage: ./scripts/run_full_pipeline.sh

set -e  # Exit on error

# Configuration
GPUS="${CUDA_VISIBLE_DEVICES:-1,2}"
NUM_GPUS=2
CONFIG="configs/training.yaml"
TRAIN_DATA="data/processed/prontoqa_train.jsonl"
VAL_DATA="data/processed/prontoqa_test.jsonl"
OAK_ITERATIONS=3

echo "============================================================"
echo "SOKRATES Full Training Pipeline"
echo "============================================================"
echo "GPUs: $GPUS"
echo "Config: $CONFIG"
echo "Train data: $TRAIN_DATA"
echo "OaK iterations: $OAK_ITERATIONS"
echo "============================================================"
echo ""

# Find the latest SFT output directory
find_latest_sft() {
    ls -td outputs/sft/*/final 2>/dev/null | head -1
}

# Step 1: SFT Training
echo "[1/3] Starting SFT Training (~1.5 hours)..."
echo "Started at: $(date)"

CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    scripts/train_sft.py \
    --config $CONFIG \
    --data $TRAIN_DATA

SFT_MODEL=$(find_latest_sft)
echo "SFT completed at: $(date)"
echo "SFT model saved to: $SFT_MODEL"
echo ""

# Step 2: OaK-DPO Loop
echo "[2/3] Starting OaK-DPO Training (~4-5 hours)..."
echo "Started at: $(date)"

CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    scripts/run_oak_dpo.py \
    --config $CONFIG \
    --sft-model "$SFT_MODEL" \
    --train-data $TRAIN_DATA \
    --val-data $VAL_DATA \
    --dataset-type prontoqa \
    --iterations $OAK_ITERATIONS

echo "OaK-DPO completed at: $(date)"
echo ""

# Step 3: Evaluation
echo "[3/3] Running Evaluation (~30 min)..."
echo "Started at: $(date)"

# Find the latest checkpoint
FINAL_MODEL=$(ls -td outputs/oak_dpo/*/checkpoints/iter_*/model 2>/dev/null | head -1)

if [ -n "$FINAL_MODEL" ]; then
    CUDA_VISIBLE_DEVICES=${GPUS%%,*} python scripts/evaluate.py \
        --model "$FINAL_MODEL" \
        --data $VAL_DATA \
        --dataset-type prontoqa \
        --output-dir outputs/evaluation \
        --dataset-name final_model
    
    echo ""
    echo "============================================================"
    echo "PIPELINE COMPLETE"
    echo "============================================================"
    echo "Final model: $FINAL_MODEL"
    echo "Results: outputs/evaluation/final_model_report.txt"
    echo "Completed at: $(date)"
    cat outputs/evaluation/final_model_report.txt
else
    echo "Warning: Could not find final model checkpoint"
fi

