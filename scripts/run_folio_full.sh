#!/bin/bash
# FOLIO: Full SFT + 3 OaK-DPO iterations + evaluation
# Usage: ./scripts/run_folio_full.sh

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "FOLIO: Full Pipeline (SFT + 3 DPO iterations)"
echo "============================================================"

# Settings
GPUS="2,3,4,5,6,7"
DATA="data/processed/folio_train.jsonl"
TEST="data/processed/folio_validation.jsonl"
NUM_PROBLEMS=1001
TEMP=0.5
MAX_STEPS=15
SAMPLES=2

# FOLIO is smaller, use 2 GPUs for trace gen (500 problems each)
PPG=$((NUM_PROBLEMS / 2 + 1))

# Helper function to get usable model path
get_model_path() {
    local dir=$1
    if [ -f "${dir}/adapter_config.json" ]; then
        # It's a LoRA adapter, check for merged
        local merged="${dir%/final}/merged"
        if [ ! -d "$merged" ]; then
            echo "  Merging LoRA adapter..." >&2
            python scripts/merge_lora_adapter.py --adapter "$dir" --output "$merged"
        fi
        echo "$merged"
    else
        # It's a full model
        echo "$dir"
    fi
}

############################################################
# SFT TRAINING
############################################################
echo ""
echo "============================================================"
echo "SFT TRAINING (FOLIO)"
echo "============================================================"

if [ ! -d "outputs/sft_folio" ] || [ ! -d "outputs/sft_folio/latest" ]; then
    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=6 --mixed_precision=bf16 \
        scripts/train_sft.py \
        --config configs/training.yaml \
        --data-path $DATA \
        --output-dir outputs/sft_folio
fi

# Get SFT model path (merge if needed)
echo "[SFT] Preparing model for inference..."
MODEL_SFT=$(get_model_path "outputs/sft_folio/latest/final")
echo "  Using: $MODEL_SFT"

############################################################
# ITERATION 1
############################################################
echo ""
echo "============================================================"
echo "ITERATION 1 (FOLIO)"
echo "============================================================"

echo "[Iter1] Generating traces (2 GPUs)..."
mkdir -p outputs/traces/folio_iter0

CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
    --model $MODEL_SFT --data $DATA \
    --output outputs/traces/folio_iter0/gpu2 \
    --num-problems $PPG --start-idx 0 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=3 python scripts/generate_traces_vllm.py \
    --model $MODEL_SFT --data $DATA \
    --output outputs/traces/folio_iter0/gpu3 \
    --num-problems $PPG --start-idx $PPG \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

wait
cat outputs/traces/folio_iter0/gpu*/traces.jsonl > outputs/traces/folio_iter0/traces.jsonl
echo "[Iter1] Traces: $(wc -l < outputs/traces/folio_iter0/traces.jsonl)"

echo "[Iter1] DPO training..."
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/folio_iter0/traces.jsonl \
    --model $MODEL_SFT \
    --output outputs/dpo/folio_iter1 \
    --num-epochs 1 --batch-size 4 --gradient-accumulation-steps 2 \
    --learning-rate 5e-6 --beta 0.1

echo "[Iter1] Complete!"

############################################################
# ITERATION 2
############################################################
echo ""
echo "============================================================"
echo "ITERATION 2 (FOLIO)"
echo "============================================================"

MODEL_ITER1=$(get_model_path "outputs/dpo/folio_iter1/final")
echo "  Using: $MODEL_ITER1"

echo "[Iter2] Generating traces..."
mkdir -p outputs/traces/folio_iter1

CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER1 --data $DATA \
    --output outputs/traces/folio_iter1/gpu2 \
    --num-problems $PPG --start-idx 0 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=3 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER1 --data $DATA \
    --output outputs/traces/folio_iter1/gpu3 \
    --num-problems $PPG --start-idx $PPG \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

wait
cat outputs/traces/folio_iter1/gpu*/traces.jsonl > outputs/traces/folio_iter1/traces.jsonl
echo "[Iter2] Traces: $(wc -l < outputs/traces/folio_iter1/traces.jsonl)"

echo "[Iter2] DPO training..."
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/folio_iter1/traces.jsonl \
    --model $MODEL_ITER1 \
    --output outputs/dpo/folio_iter2 \
    --num-epochs 1 --batch-size 4 --gradient-accumulation-steps 2 \
    --learning-rate 5e-6 --beta 0.1

echo "[Iter2] Complete!"

############################################################
# ITERATION 3
############################################################
echo ""
echo "============================================================"
echo "ITERATION 3 (FOLIO)"
echo "============================================================"

MODEL_ITER2=$(get_model_path "outputs/dpo/folio_iter2/final")
echo "  Using: $MODEL_ITER2"

echo "[Iter3] Generating traces..."
mkdir -p outputs/traces/folio_iter2

CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER2 --data $DATA \
    --output outputs/traces/folio_iter2/gpu2 \
    --num-problems $PPG --start-idx 0 \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

CUDA_VISIBLE_DEVICES=3 python scripts/generate_traces_vllm.py \
    --model $MODEL_ITER2 --data $DATA \
    --output outputs/traces/folio_iter2/gpu3 \
    --num-problems $PPG --start-idx $PPG \
    --samples-per-problem $SAMPLES --temperature $TEMP --max-steps $MAX_STEPS &

wait
cat outputs/traces/folio_iter2/gpu*/traces.jsonl > outputs/traces/folio_iter2/traces.jsonl
echo "[Iter3] Traces: $(wc -l < outputs/traces/folio_iter2/traces.jsonl)"

echo "[Iter3] DPO training..."
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/folio_iter2/traces.jsonl \
    --model $MODEL_ITER2 \
    --output outputs/dpo/folio_iter3 \
    --num-epochs 1 --batch-size 4 --gradient-accumulation-steps 2 \
    --learning-rate 5e-6 --beta 0.1

echo "[Iter3] Complete!"

############################################################
# FINAL EVALUATION
############################################################
echo ""
echo "============================================================"
echo "FINAL EVALUATION (FOLIO)"
echo "============================================================"

MODEL_FINAL=$(get_model_path "outputs/dpo/folio_iter3/final")
echo "  Using: $MODEL_FINAL"

echo "[Eval] Evaluating on validation set..."
CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
    --model $MODEL_FINAL \
    --data $TEST \
    --output outputs/eval/folio_final \
    --num-problems 0 \
    --samples-per-problem 1 \
    --temperature 0.0

echo ""
echo "============================================================"
echo "FOLIO COMPLETE!"
echo "============================================================"
echo "Final model: $MODEL_FINAL"
echo "Evaluation: outputs/eval/folio_final/traces.jsonl"
echo "============================================================"
