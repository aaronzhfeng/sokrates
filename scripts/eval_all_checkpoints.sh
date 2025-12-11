#!/bin/bash
# Evaluate ALL our checkpoints (7 total) using 6 GPUs DATA-PARALLEL per task
# Each model evaluation is split across 6 GPUs for maximum speed
#
# Usage: ./scripts/eval_all_checkpoints.sh

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "EVALUATING ALL SOKRATES CHECKPOINTS (6 GPU DATA-PARALLEL)"
echo "============================================================"

# Output directory for all evaluations
EVAL_DIR="outputs/eval/all_checkpoints"
mkdir -p $EVAL_DIR

# Test datasets
PRONTOQA_TEST="data/processed/prontoqa_test.jsonl"
FOLIO_TEST="data/processed/folio_validation.jsonl"

# GPU list
GPUS=(2 3 4 5 6 7)
NUM_GPUS=6

# Get problem counts
PRONTOQA_COUNT=$(wc -l < $PRONTOQA_TEST)
FOLIO_COUNT=$(wc -l < $FOLIO_TEST)
echo "PrOntoQA test: $PRONTOQA_COUNT problems"
echo "FOLIO validation: $FOLIO_COUNT problems"

# Problems per GPU
PPG_PRONTOQA=$(( (PRONTOQA_COUNT + NUM_GPUS - 1) / NUM_GPUS ))
PPG_FOLIO=$(( (FOLIO_COUNT + NUM_GPUS - 1) / NUM_GPUS ))

# Helper function
get_model_path() {
    local dir=$1
    if [ -f "${dir}/adapter_config.json" ]; then
        local merged="${dir%/final}/merged"
        if [ ! -d "$merged" ]; then
            echo "  Merging LoRA adapter..." >&2
            python scripts/merge_lora_adapter.py --adapter "$dir" --output "$merged"
        fi
        echo "$merged"
    else
        echo "$dir"
    fi
}

# Function to run evaluation on 6 GPUs in parallel with visible progress
run_eval_parallel() {
    local MODEL_PATH=$1
    local DATA=$2
    local OUTPUT_DIR=$3
    local NAME=$4
    local TOTAL_PROBLEMS=$5
    local PPG=$6
    
    echo ""
    echo "============================================================"
    echo "[Eval] $NAME - 6 GPUs × ~$PPG problems each"
    echo "============================================================"
    mkdir -p "$OUTPUT_DIR"
    
    # Launch 6 parallel processes - output visible with tqdm
    for i in 0 1 2 3 4 5; do
        GPU_ID=${GPUS[$i]}
        START=$((i * PPG))
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/generate_traces_vllm.py \
            --model "$MODEL_PATH" \
            --data "$DATA" \
            --output "$OUTPUT_DIR/gpu${GPU_ID}" \
            --num-problems $PPG \
            --start-idx $START \
            --samples-per-problem 1 \
            --temperature 0.0 &
    done
    
    # Wait for all to complete
    wait
    
    # Merge results
    echo ""
    echo "Merging results from 6 GPUs..."
    cat "$OUTPUT_DIR"/gpu*/traces.jsonl > "$OUTPUT_DIR/traces.jsonl" 2>/dev/null || true
    
    # Create combined summary
    python3 << EOF
import json
from pathlib import Path

output_dir = Path("$OUTPUT_DIR")
total_correct = 0
total_problems = 0
total_valid_traces = 0

# For weighted average of step_validity
weighted_step_validity = 0.0

for gpu_dir in sorted(output_dir.glob("gpu*")):
    summary_file = gpu_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            d = json.load(f)
            n = d.get("total_traces", 0)
            total_correct += d.get("correct_traces", 0)
            total_problems += n
            total_valid_traces += d.get("valid_traces", 0)
            # Weight step_validity by number of traces
            weighted_step_validity += d.get("step_validity", 0) * n

accuracy = total_correct / total_problems if total_problems > 0 else 0
step_validity = weighted_step_validity / total_problems if total_problems > 0 else 0
trace_validity = total_valid_traces / total_problems if total_problems > 0 else 0

summary = {
    "model": "$NAME",
    "total_traces": total_problems,
    "correct_traces": total_correct,
    "accuracy": accuracy,
    "step_validity": step_validity,
    "valid_traces": total_valid_traces,
    "trace_validity": trace_validity
}

with open(output_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"")
print(f"============================================================")
print(f"✓ {summary['model']} COMPLETE")
print(f"  Accuracy:       {accuracy:.1%} ({total_correct}/{total_problems})")
print(f"  Step Validity:  {step_validity:.1%}")  
print(f"  Trace Validity: {trace_validity:.1%}")
print(f"============================================================")
EOF
}

############################################################
# PREPARE ALL MODELS
############################################################
echo ""
echo "Preparing models (merging LoRA if needed)..."

declare -A MODELS
declare -A MODEL_NAMES

# PrOntoQA models
[ -d "outputs/sft/latest/final" ] && MODELS[prontoqa_sft]=$(get_model_path "outputs/sft/latest/final") && MODEL_NAMES[prontoqa_sft]="PrOntoQA-SFT"
[ -d "outputs/dpo/iter1/final" ] && MODELS[prontoqa_dpo_iter1]=$(get_model_path "outputs/dpo/iter1/final") && MODEL_NAMES[prontoqa_dpo_iter1]="PrOntoQA-DPO-iter1"
[ -d "outputs/dpo/iter2/final" ] && MODELS[prontoqa_dpo_iter2]=$(get_model_path "outputs/dpo/iter2/final") && MODEL_NAMES[prontoqa_dpo_iter2]="PrOntoQA-DPO-iter2"
[ -d "outputs/dpo/iter3/final" ] && MODELS[prontoqa_dpo_iter3]=$(get_model_path "outputs/dpo/iter3/final") && MODEL_NAMES[prontoqa_dpo_iter3]="PrOntoQA-DPO-iter3"

# FOLIO models
[ -d "outputs/dpo/folio_iter1/final" ] && MODELS[folio_dpo_iter1]=$(get_model_path "outputs/dpo/folio_iter1/final") && MODEL_NAMES[folio_dpo_iter1]="FOLIO-DPO-iter1"
[ -d "outputs/dpo/folio_iter2/final" ] && MODELS[folio_dpo_iter2]=$(get_model_path "outputs/dpo/folio_iter2/final") && MODEL_NAMES[folio_dpo_iter2]="FOLIO-DPO-iter2"
[ -d "outputs/dpo/folio_iter3/final" ] && MODELS[folio_dpo_iter3]=$(get_model_path "outputs/dpo/folio_iter3/final") && MODEL_NAMES[folio_dpo_iter3]="FOLIO-DPO-iter3"

echo "Found ${#MODELS[@]} models to evaluate"

############################################################
# PrOntoQA EVALUATIONS (4 models on PrOntoQA test)
############################################################
echo ""
echo "########################################################"
echo "# PrOntoQA Evaluations (4 models)"
echo "########################################################"

for key in prontoqa_sft prontoqa_dpo_iter1 prontoqa_dpo_iter2 prontoqa_dpo_iter3; do
    if [ -n "${MODELS[$key]}" ]; then
        run_eval_parallel "${MODELS[$key]}" "$PRONTOQA_TEST" "$EVAL_DIR/$key" "${MODEL_NAMES[$key]}" $PRONTOQA_COUNT $PPG_PRONTOQA
    fi
done

############################################################
# FOLIO EVALUATIONS (3 models on FOLIO validation)
############################################################
echo ""
echo "########################################################"
echo "# FOLIO Evaluations (3 models)"
echo "########################################################"

for key in folio_dpo_iter1 folio_dpo_iter2 folio_dpo_iter3; do
    if [ -n "${MODELS[$key]}" ]; then
        run_eval_parallel "${MODELS[$key]}" "$FOLIO_TEST" "$EVAL_DIR/$key" "${MODEL_NAMES[$key]}" $FOLIO_COUNT $PPG_FOLIO
    fi
done

############################################################
# FINAL SUMMARY
############################################################

echo ""
echo ""
echo "########################################################"
echo "# FINAL SUMMARY - ALL CHECKPOINTS"
echo "########################################################"

echo ""
echo "PrOntoQA Test Results:"
echo "----------------------"
for model in prontoqa_sft prontoqa_dpo_iter1 prontoqa_dpo_iter2 prontoqa_dpo_iter3; do
    if [ -f "$EVAL_DIR/${model}/summary.json" ]; then
        python3 -c "
import json
with open('$EVAL_DIR/${model}/summary.json') as f:
    d = json.load(f)
    acc = d.get('accuracy', 0)
    sv = d.get('step_validity', 0)
    tv = d.get('trace_validity', 0)
    print(f'{d[\"model\"]:25} | Acc: {acc:6.1%} | StepVal: {sv:6.1%} | TraceVal: {tv:6.1%}')
"
    fi
done

echo ""
echo "FOLIO Validation Results:"
echo "-------------------------"
for model in folio_dpo_iter1 folio_dpo_iter2 folio_dpo_iter3; do
    if [ -f "$EVAL_DIR/${model}/summary.json" ]; then
        python3 -c "
import json
with open('$EVAL_DIR/${model}/summary.json') as f:
    d = json.load(f)
    acc = d.get('accuracy', 0)
    sv = d.get('step_validity', 0)
    tv = d.get('trace_validity', 0)
    print(f'{d[\"model\"]:25} | Acc: {acc:6.1%} | StepVal: {sv:6.1%} | TraceVal: {tv:6.1%}')
"
    fi
done

echo ""
echo "All results saved to: $EVAL_DIR/"
