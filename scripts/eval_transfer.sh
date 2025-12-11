#!/bin/bash
# Evaluate TRANSFER performance (6 GPU DATA-PARALLEL per task)
# - PrOntoQA models on FOLIO (zero-shot transfer)
# - FOLIO models on PrOntoQA (backward transfer)
#
# Usage: ./scripts/eval_transfer.sh

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "TRANSFER EVALUATION (6 GPU DATA-PARALLEL)"
echo "============================================================"
echo ""
echo "Testing cross-dataset generalization:"
echo "  - PrOntoQA SFT → FOLIO"
echo "  - PrOntoQA DPO iter3 → FOLIO"
echo "  - FOLIO DPO iter3 → PrOntoQA"
echo "============================================================"

EVAL_DIR="outputs/eval/transfer"
mkdir -p $EVAL_DIR

PRONTOQA_TEST="data/processed/prontoqa_test.jsonl"
FOLIO_TEST="data/processed/folio_validation.jsonl"

# GPU list
GPUS=(2 3 4 5 6 7)
NUM_GPUS=6

# Problem counts
PRONTOQA_COUNT=$(wc -l < $PRONTOQA_TEST)
FOLIO_COUNT=$(wc -l < $FOLIO_TEST)

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

# Function to run transfer evaluation with 6 GPUs
run_transfer_parallel() {
    local MODEL_PATH=$1
    local DATA=$2
    local OUTPUT_DIR=$3
    local NAME=$4
    local TOTAL=$5
    local PPG=$6
    
    echo ""
    echo "============================================================"
    echo "[Transfer] $NAME - 6 GPUs × ~$PPG problems each"
    echo "============================================================"
    mkdir -p "$OUTPUT_DIR"
    
    # Launch 6 parallel processes
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
            weighted_step_validity += d.get("step_validity", 0) * n

accuracy = total_correct / total_problems if total_problems > 0 else 0
step_validity = weighted_step_validity / total_problems if total_problems > 0 else 0
trace_validity = total_valid_traces / total_problems if total_problems > 0 else 0

summary = {
    "transfer": "$NAME",
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
print(f"✓ $NAME COMPLETE")
print(f"  Accuracy:       {accuracy:.1%} ({total_correct}/{total_problems})")
print(f"  Step Validity:  {step_validity:.1%}")
print(f"  Trace Validity: {trace_validity:.1%}")
print(f"============================================================")
EOF
}

############################################################
# PREPARE MODELS
############################################################
echo ""
echo "Preparing models..."

MODEL_SFT=""
MODEL_PRONTOQA_DPO3=""
MODEL_FOLIO_DPO3=""

[ -d "outputs/sft/latest/final" ] && MODEL_SFT=$(get_model_path "outputs/sft/latest/final")
[ -d "outputs/dpo/iter3/final" ] && MODEL_PRONTOQA_DPO3=$(get_model_path "outputs/dpo/iter3/final")
[ -d "outputs/dpo/folio_iter3/final" ] && MODEL_FOLIO_DPO3=$(get_model_path "outputs/dpo/folio_iter3/final")

############################################################
# TRANSFER EVALUATIONS
############################################################

echo ""
echo "########################################################"
echo "# PrOntoQA → FOLIO (Zero-shot Transfer)"
echo "########################################################"

# PrOntoQA SFT → FOLIO
if [ -n "$MODEL_SFT" ]; then
    run_transfer_parallel "$MODEL_SFT" "$FOLIO_TEST" "$EVAL_DIR/prontoqa_sft_to_folio" "PrOntoQA-SFT → FOLIO" $FOLIO_COUNT $PPG_FOLIO
fi

# PrOntoQA DPO iter3 → FOLIO
if [ -n "$MODEL_PRONTOQA_DPO3" ]; then
    run_transfer_parallel "$MODEL_PRONTOQA_DPO3" "$FOLIO_TEST" "$EVAL_DIR/prontoqa_dpo_iter3_to_folio" "PrOntoQA-DPO-iter3 → FOLIO" $FOLIO_COUNT $PPG_FOLIO
fi

echo ""
echo "########################################################"
echo "# FOLIO → PrOntoQA (Backward Transfer)"
echo "########################################################"

# FOLIO DPO iter3 → PrOntoQA
if [ -n "$MODEL_FOLIO_DPO3" ]; then
    run_transfer_parallel "$MODEL_FOLIO_DPO3" "$PRONTOQA_TEST" "$EVAL_DIR/folio_dpo_iter3_to_prontoqa" "FOLIO-DPO-iter3 → PrOntoQA" $PRONTOQA_COUNT $PPG_PRONTOQA
fi

############################################################
# FINAL SUMMARY
############################################################

echo ""
echo ""
echo "########################################################"
echo "# FINAL SUMMARY - TRANSFER EVALUATION"
echo "########################################################"

echo ""
echo "PrOntoQA → FOLIO:"
echo "-----------------"
for name in prontoqa_sft_to_folio prontoqa_dpo_iter3_to_folio; do
    if [ -f "$EVAL_DIR/$name/summary.json" ]; then
        python3 -c "
import json
with open('$EVAL_DIR/$name/summary.json') as f:
    d = json.load(f)
    print(f'  {d[\"transfer\"]:35} | Acc: {d[\"accuracy\"]:6.1%} | StepVal: {d[\"step_validity\"]:6.1%}')
"
    fi
done

echo ""
echo "FOLIO → PrOntoQA:"
echo "-----------------"
if [ -f "$EVAL_DIR/folio_dpo_iter3_to_prontoqa/summary.json" ]; then
    python3 -c "
import json
with open('$EVAL_DIR/folio_dpo_iter3_to_prontoqa/summary.json') as f:
    d = json.load(f)
    print(f'  {d[\"transfer\"]:35} | Acc: {d[\"accuracy\"]:6.1%} | StepVal: {d[\"step_validity\"]:6.1%}')
"
fi

echo ""
echo "All transfer results saved to: $EVAL_DIR/"
