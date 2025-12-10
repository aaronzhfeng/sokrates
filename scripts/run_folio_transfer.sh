#!/bin/bash
# ==============================================================================
# FOLIO Transfer Learning Experiment
# ==============================================================================
# This script demonstrates cross-dataset transfer and domain adaptation:
# 1. Zero-shot evaluation: PrOntoQA SFT model on FOLIO
# 2. Domain adaptation: 3 iterations of DPO on FOLIO
# 3. (Optional) Ablation: Base LLM → DPO on FOLIO (no SFT)
#
# Usage: ./scripts/run_folio_transfer.sh [--with-ablation]
# ==============================================================================

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

# Configuration
GPUS="2,3,4,5,6,7"
N_GPUS=6
FOLIO_TRAIN="data/processed/folio_train.jsonl"
FOLIO_VAL="data/processed/folio_validation.jsonl"
N_TRAIN=1001
N_VAL=203
TEMP=0.5
MAX_STEPS=15
SAMPLES=2

# Model paths
SFT_MODEL="outputs/sft/latest/merged"
BASE_MODEL="Qwen/Qwen3-8B"

# Output directories
OUTPUT_DIR="outputs/folio_transfer"
TRACES_DIR="${OUTPUT_DIR}/traces"
DPO_DIR="${OUTPUT_DIR}/dpo"
EVAL_DIR="${OUTPUT_DIR}/eval"

# Check if ablation requested
WITH_ABLATION=false
if [[ "$1" == "--with-ablation" ]]; then
    WITH_ABLATION=true
    echo "Will run ablation experiment (Base LLM → DPO)"
fi

echo "============================================================"
echo "FOLIO Transfer Learning Experiment"
echo "============================================================"
echo "SFT Model: ${SFT_MODEL}"
echo "FOLIO Train: ${N_TRAIN} problems"
echo "FOLIO Val: ${N_VAL} problems"
echo "GPUs: ${GPUS}"
echo "============================================================"

# Create output directories
mkdir -p ${TRACES_DIR} ${DPO_DIR} ${EVAL_DIR}

# ==============================================================================
# STAGE 0: Zero-Shot Evaluation
# ==============================================================================
echo ""
echo "============================================================"
echo "[Stage 0] Zero-Shot Evaluation: PrOntoQA SFT → FOLIO"
echo "============================================================"

CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
    --model ${SFT_MODEL} \
    --data ${FOLIO_VAL} \
    --output ${EVAL_DIR}/zero_shot \
    --num-problems 0 \
    --samples-per-problem 1 \
    --temperature 0.0 \
    --max-steps ${MAX_STEPS}

echo ""
echo "Zero-shot results saved to: ${EVAL_DIR}/zero_shot/"

# ==============================================================================
# STAGE 1: Generate Traces on FOLIO Train
# ==============================================================================
echo ""
echo "============================================================"
echo "[Stage 1] Generating Traces on FOLIO Train (${N_TRAIN} problems)"
echo "============================================================"

# Calculate problems per GPU
PPG=$((N_TRAIN / N_GPUS + 1))

# Run parallel trace generation
echo "Launching ${N_GPUS} parallel vLLM processes..."

GPU_ARRAY=(2 3 4 5 6 7)
PIDS=()

for i in "${!GPU_ARRAY[@]}"; do
    GPU=${GPU_ARRAY[$i]}
    START=$((i * PPG))
    
    # Skip if start is beyond total problems
    if [ $START -ge $N_TRAIN ]; then
        continue
    fi
    
    mkdir -p ${TRACES_DIR}/iter0_gpu${i}
    
    echo "  GPU ${GPU}: problems ${START} to $((START + PPG - 1))"
    
    CUDA_VISIBLE_DEVICES=${GPU} python scripts/generate_traces_vllm.py \
        --model ${SFT_MODEL} \
        --data ${FOLIO_TRAIN} \
        --output ${TRACES_DIR}/iter0_gpu${i} \
        --start-idx ${START} \
        --num-problems ${PPG} \
        --samples-per-problem ${SAMPLES} \
        --temperature ${TEMP} \
        --max-steps ${MAX_STEPS} \
        > ${TRACES_DIR}/iter0_gpu${i}/log.txt 2>&1 &
    
    PIDS+=($!)
done

# Wait for all to complete
echo ""
echo "Waiting for trace generation to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

# Combine traces
echo "Combining traces from all GPUs..."
cat ${TRACES_DIR}/iter0_gpu*/traces.jsonl > ${TRACES_DIR}/iter0_all.jsonl
N_TRACES=$(wc -l < ${TRACES_DIR}/iter0_all.jsonl)
echo "Total traces: ${N_TRACES}"

# ==============================================================================
# STAGE 2: DPO Iterations
# ==============================================================================
CURRENT_MODEL=${SFT_MODEL}

for ITER in 1 2 3; do
    echo ""
    echo "============================================================"
    echo "[Stage 2.${ITER}] DPO Iteration ${ITER}"
    echo "============================================================"
    
    PREV_ITER=$((ITER - 1))
    TRACE_FILE=${TRACES_DIR}/iter${PREV_ITER}_all.jsonl
    
    echo "Training DPO on traces: ${TRACE_FILE}"
    echo "Base model: ${CURRENT_MODEL}"
    
    # DPO Training
    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --num_processes ${N_GPUS} \
        --mixed_precision bf16 \
        scripts/train_dpo_from_traces.py \
        --traces ${TRACE_FILE} \
        --model ${CURRENT_MODEL} \
        --output ${DPO_DIR}/iter${ITER} \
        --epochs 1 \
        --batch-size 4 \
        --learning-rate 5e-6 \
        --beta 0.1
    
    # Update current model
    CURRENT_MODEL=${DPO_DIR}/iter${ITER}/final
    
    # Generate traces for next iteration (if not last)
    if [ ${ITER} -lt 3 ]; then
        echo ""
        echo "Generating traces for iteration $((ITER + 1))..."
        
        PIDS=()
        for i in "${!GPU_ARRAY[@]}"; do
            GPU=${GPU_ARRAY[$i]}
            START=$((i * PPG))
            
            if [ $START -ge $N_TRAIN ]; then
                continue
            fi
            
            mkdir -p ${TRACES_DIR}/iter${ITER}_gpu${i}
            
            CUDA_VISIBLE_DEVICES=${GPU} python scripts/generate_traces_vllm.py \
                --model ${CURRENT_MODEL} \
                --data ${FOLIO_TRAIN} \
                --output ${TRACES_DIR}/iter${ITER}_gpu${i} \
                --start-idx ${START} \
                --num-problems ${PPG} \
                --samples-per-problem ${SAMPLES} \
                --temperature ${TEMP} \
                --max-steps ${MAX_STEPS} \
                > ${TRACES_DIR}/iter${ITER}_gpu${i}/log.txt 2>&1 &
            
            PIDS+=($!)
        done
        
        for pid in "${PIDS[@]}"; do
            wait $pid
        done
        
        cat ${TRACES_DIR}/iter${ITER}_gpu*/traces.jsonl > ${TRACES_DIR}/iter${ITER}_all.jsonl
    fi
done

# ==============================================================================
# STAGE 3: Final Evaluation
# ==============================================================================
echo ""
echo "============================================================"
echo "[Stage 3] Final Evaluation on FOLIO Validation"
echo "============================================================"

CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
    --model ${CURRENT_MODEL} \
    --data ${FOLIO_VAL} \
    --output ${EVAL_DIR}/final \
    --num-problems 0 \
    --samples-per-problem 1 \
    --temperature 0.0 \
    --max-steps ${MAX_STEPS}

# ==============================================================================
# STAGE 4: Ablation (Optional)
# ==============================================================================
if [ "$WITH_ABLATION" = true ]; then
    echo ""
    echo "============================================================"
    echo "[Stage 4] Ablation: Base LLM → DPO (No SFT)"
    echo "============================================================"
    
    ABLATION_DIR="${OUTPUT_DIR}/ablation_no_sft"
    mkdir -p ${ABLATION_DIR}
    
    # Generate traces with base model
    echo "Generating traces with base model..."
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
        --model ${BASE_MODEL} \
        --data ${FOLIO_TRAIN} \
        --output ${ABLATION_DIR}/traces_iter0 \
        --num-problems ${N_TRAIN} \
        --samples-per-problem ${SAMPLES} \
        --temperature ${TEMP} \
        --max-steps ${MAX_STEPS}
    
    # 3 DPO iterations from base
    ABLATION_MODEL=${BASE_MODEL}
    for ITER in 1 2 3; do
        echo "Ablation DPO iteration ${ITER}..."
        
        PREV_ITER=$((ITER - 1))
        
        CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
            --num_processes ${N_GPUS} \
            --mixed_precision bf16 \
            scripts/train_dpo_from_traces.py \
            --traces ${ABLATION_DIR}/traces_iter${PREV_ITER}/traces.jsonl \
            --model ${ABLATION_MODEL} \
            --output ${ABLATION_DIR}/dpo_iter${ITER} \
            --epochs 1 \
            --batch-size 4 \
            --learning-rate 5e-6 \
            --beta 0.1
        
        ABLATION_MODEL=${ABLATION_DIR}/dpo_iter${ITER}/final
        
        if [ ${ITER} -lt 3 ]; then
            CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
                --model ${ABLATION_MODEL} \
                --data ${FOLIO_TRAIN} \
                --output ${ABLATION_DIR}/traces_iter${ITER} \
                --num-problems ${N_TRAIN} \
                --samples-per-problem ${SAMPLES} \
                --temperature ${TEMP} \
                --max-steps ${MAX_STEPS}
        fi
    done
    
    # Evaluate ablation
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces_vllm.py \
        --model ${ABLATION_MODEL} \
        --data ${FOLIO_VAL} \
        --output ${ABLATION_DIR}/eval_final \
        --num-problems 0 \
        --samples-per-problem 1 \
        --temperature 0.0 \
        --max-steps ${MAX_STEPS}
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "============================================================"
echo "FOLIO Transfer Experiment Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Zero-shot eval: ${EVAL_DIR}/zero_shot/summary.json"
echo "  Final eval:     ${EVAL_DIR}/final/summary.json"
if [ "$WITH_ABLATION" = true ]; then
    echo "  Ablation eval:  ${ABLATION_DIR}/eval_final/summary.json"
fi
echo ""
echo "Compare results:"
echo "  cat ${EVAL_DIR}/zero_shot/summary.json"
echo "  cat ${EVAL_DIR}/final/summary.json"
echo ""

# Print summary comparison
echo "Quick Summary:"
echo "─────────────────────────────────────────────────────────────"
if [ -f "${EVAL_DIR}/zero_shot/summary.json" ]; then
    ZERO_ACC=$(python3 -c "import json; d=json.load(open('${EVAL_DIR}/zero_shot/summary.json')); print(f\"{d.get('accuracy', d.get('correct_rate', 0))*100:.1f}%\")")
    echo "  Zero-shot (PrOntoQA SFT → FOLIO):  ${ZERO_ACC}"
fi
if [ -f "${EVAL_DIR}/final/summary.json" ]; then
    FINAL_ACC=$(python3 -c "import json; d=json.load(open('${EVAL_DIR}/final/summary.json')); print(f\"{d.get('accuracy', d.get('correct_rate', 0))*100:.1f}%\")")
    echo "  After DPO (FOLIO-adapted):         ${FINAL_ACC}"
fi
if [ "$WITH_ABLATION" = true ] && [ -f "${ABLATION_DIR}/eval_final/summary.json" ]; then
    ABLATION_ACC=$(python3 -c "import json; d=json.load(open('${ABLATION_DIR}/eval_final/summary.json')); print(f\"{d.get('accuracy', d.get('correct_rate', 0))*100:.1f}%\")")
    echo "  Ablation (Base → DPO, no SFT):     ${ABLATION_ACC}"
fi
echo "─────────────────────────────────────────────────────────────"

