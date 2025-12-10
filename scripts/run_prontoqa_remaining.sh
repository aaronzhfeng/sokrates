#!/bin/bash
# PrOntoQA: Complete iterations 2-3 + final evaluation
# Usage: ./scripts/run_prontoqa_remaining.sh

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "PrOntoQA OaK-DPO: Iterations 2-3 + Evaluation"
echo "============================================================"

# Settings
GPUS="2,3,4,5,6,7"
DATA="data/processed/prontoqa_train.jsonl"
TEST="data/processed/prontoqa_test.jsonl"
TEMP=0.5
MAX_STEPS=15
SAMPLES=2

# Problems per GPU (14346 / 6 = 2391)
PPG=2391

############################################################
# ITERATION 2
############################################################
echo ""
echo "============================================================"
echo "ITERATION 2"
echo "============================================================"

# Step 2.1: Check if DPO iter1 is full model or LoRA
echo "[Iter2] Checking DPO iter1 model type..."
if [ -f "outputs/dpo/iter1/final/adapter_config.json" ]; then
    # It's a LoRA adapter, need to merge
    if [ ! -d "outputs/dpo/iter1/merged" ]; then
        echo "  Merging LoRA adapter..."
        python scripts/merge_lora_adapter.py \
            --adapter outputs/dpo/iter1/final \
            --output outputs/dpo/iter1/merged
    fi
    MODEL_ITER1="outputs/dpo/iter1/merged"
else
    # It's already a full model, use directly
    echo "  Already a full model, no merge needed"
    MODEL_ITER1="outputs/dpo/iter1/final"
fi

# Step 2.2: Generate traces
echo "[Iter2] Generating traces (6 GPUs)..."
mkdir -p outputs/traces/iter1

CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER1 --data $DATA \
    --output outputs/traces/iter1/gpu2 \
    --num-problems $PPG --start-idx 0 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=3 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER1 --data $DATA \
    --output outputs/traces/iter1/gpu3 \
    --num-problems $PPG --start-idx 2391 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=4 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER1 --data $DATA \
    --output outputs/traces/iter1/gpu4 \
    --num-problems $PPG --start-idx 4782 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=5 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER1 --data $DATA \
    --output outputs/traces/iter1/gpu5 \
    --num-problems $PPG --start-idx 7173 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=6 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER1 --data $DATA \
    --output outputs/traces/iter1/gpu6 \
    --num-problems $PPG --start-idx 9564 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=7 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER1 --data $DATA \
    --output outputs/traces/iter1/gpu7 \
    --num-problems $PPG --start-idx 11955 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

wait
echo "[Iter2] Merging trace files..."
cat outputs/traces/iter1/gpu*/traces.jsonl > outputs/traces/iter1/traces.jsonl
echo "[Iter2] Traces: $(wc -l < outputs/traces/iter1/traces.jsonl)"

# Step 2.3: DPO training
echo "[Iter2] DPO training..."
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter1/traces.jsonl \
    --model $MODEL_ITER1 \
    --output outputs/dpo/iter2 \
    --num-epochs 1 --batch-size 4 --gradient-accumulation-steps 2 \
    --learning-rate 5e-6 --beta 0.1

echo "[Iter2] Complete!"

############################################################
# ITERATION 3
############################################################
echo ""
echo "============================================================"
echo "ITERATION 3"
echo "============================================================"

# Step 3.1: Check if DPO iter2 is full model or LoRA
echo "[Iter3] Checking DPO iter2 model type..."
if [ -f "outputs/dpo/iter2/final/adapter_config.json" ]; then
    if [ ! -d "outputs/dpo/iter2/merged" ]; then
        echo "  Merging LoRA adapter..."
        python scripts/merge_lora_adapter.py \
            --adapter outputs/dpo/iter2/final \
            --output outputs/dpo/iter2/merged
    fi
    MODEL_ITER2="outputs/dpo/iter2/merged"
else
    echo "  Already a full model, no merge needed"
    MODEL_ITER2="outputs/dpo/iter2/final"
fi

# Step 3.2: Generate traces
echo "[Iter3] Generating traces (6 GPUs)..."
mkdir -p outputs/traces/iter2

CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER2 --data $DATA \
    --output outputs/traces/iter2/gpu2 \
    --num-problems $PPG --start-idx 0 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=3 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER2 --data $DATA \
    --output outputs/traces/iter2/gpu3 \
    --num-problems $PPG --start-idx 2391 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=4 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER2 --data $DATA \
    --output outputs/traces/iter2/gpu4 \
    --num-problems $PPG --start-idx 4782 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=5 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER2 --data $DATA \
    --output outputs/traces/iter2/gpu5 \
    --num-problems $PPG --start-idx 7173 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=6 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER2 --data $DATA \
    --output outputs/traces/iter2/gpu6 \
    --num-problems $PPG --start-idx 9564 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=7 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER2 --data $DATA \
    --output outputs/traces/iter2/gpu7 \
    --num-problems $PPG --start-idx 11955 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

wait
echo "[Iter3] Merging trace files..."
cat outputs/traces/iter2/gpu*/traces.jsonl > outputs/traces/iter2/traces.jsonl
echo "[Iter3] Traces: $(wc -l < outputs/traces/iter2/traces.jsonl)"

# Step 3.3: DPO training
echo "[Iter3] DPO training..."
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter2/traces.jsonl \
    --model $MODEL_ITER2 \
    --output outputs/dpo/iter3 \
    --num-epochs 1 --batch-size 4 --gradient-accumulation-steps 2 \
    --learning-rate 5e-6 --beta 0.1

echo "[Iter3] Complete!"

############################################################
# FINAL EVALUATION
############################################################
echo ""
echo "============================================================"
echo "FINAL EVALUATION"
echo "============================================================"

# Check if final model needs merging
echo "[Eval] Preparing final model..."
if [ -f "outputs/dpo/iter3/final/adapter_config.json" ]; then
    python scripts/merge_lora_adapter.py \
        --adapter outputs/dpo/iter3/final \
        --output outputs/dpo/iter3/merged
    MODEL_FINAL="outputs/dpo/iter3/merged"
else
    MODEL_FINAL="outputs/dpo/iter3/final"
fi

# Evaluate on test set
echo "[Eval] Evaluating on test set..."
CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
    --model $MODEL_FINAL \
    --data $TEST \
    --output outputs/eval/prontoqa_final \
    --num-problems 0 \
    --samples-per-problem 1 \
    --temperature 0.0

echo ""
echo "============================================================"
echo "PrOntoQA COMPLETE!"
echo "============================================================"
echo "Final model: outputs/dpo/iter3/merged"
echo "Evaluation: outputs/eval/prontoqa_final/traces.jsonl"
echo "============================================================"

