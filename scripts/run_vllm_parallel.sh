#!/bin/bash
# Parallel vLLM trace generation across 6 GPUs (data parallel)
# 
# Usage: ./scripts/run_vllm_parallel.sh [output_dir] [num_problems]
# Example: ./scripts/run_vllm_parallel.sh outputs/traces/iter0 14346

set -e

OUTPUT_BASE=${1:-outputs/traces/iter0}
TOTAL_PROBLEMS=${2:-14346}
MODEL="outputs/sft/latest/merged"
DATA="data/processed/prontoqa_train.jsonl"
SAMPLES=2
TEMP=0.5
MAX_STEPS=15

# Calculate problems per GPU (6 GPUs: 2-7)
PROBLEMS_PER_GPU=$((TOTAL_PROBLEMS / 6))
REMAINDER=$((TOTAL_PROBLEMS % 6))

echo "============================================================"
echo "Parallel vLLM Trace Generation"
echo "============================================================"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Total problems: $TOTAL_PROBLEMS"
echo "Problems per GPU: ~$PROBLEMS_PER_GPU"
echo "Output: $OUTPUT_BASE"
echo "============================================================"

mkdir -p $OUTPUT_BASE

# Start 6 processes on GPUs 2-7
START_IDX=0
PIDS=()

for GPU_ID in 2 3 4 5 6 7; do
    # Last GPU gets remainder
    if [ $GPU_ID -eq 7 ]; then
        N_PROBLEMS=$((PROBLEMS_PER_GPU + REMAINDER))
    else
        N_PROBLEMS=$PROBLEMS_PER_GPU
    fi
    
    OUTPUT_DIR="${OUTPUT_BASE}/gpu${GPU_ID}"
    SEED=$((42 + GPU_ID))
    
    echo "[GPU $GPU_ID] Processing $N_PROBLEMS problems (start: $START_IDX)"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/generate_traces_vllm.py \
        --model $MODEL \
        --data $DATA \
        --output $OUTPUT_DIR \
        --num-problems $N_PROBLEMS \
        --start-idx $START_IDX \
        --samples-per-problem $SAMPLES \
        --temperature $TEMP \
        --max-steps $MAX_STEPS \
        --seed $SEED \
        > "${OUTPUT_DIR}.log" 2>&1 &
    
    PIDS+=($!)
    START_IDX=$((START_IDX + N_PROBLEMS))
done

echo ""
echo "Started ${#PIDS[@]} processes: ${PIDS[@]}"
echo "Logs: ${OUTPUT_BASE}/gpu*.log"
echo ""

# Monitor progress while waiting
echo "Progress (updates every 30s):"
while true; do
    # Check if any process still running
    RUNNING=0
    for PID in ${PIDS[@]}; do
        if kill -0 $PID 2>/dev/null; then
            RUNNING=1
            break
        fi
    done
    
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    
    # Show progress from each GPU
    echo -n "$(date +%H:%M:%S) | "
    for GPU_ID in 2 3 4 5 6 7; do
        LOG="${OUTPUT_BASE}/gpu${GPU_ID}.log"
        if [ -f "$LOG" ]; then
            # Extract latest progress percentage
            PROGRESS=$(grep -oP '\d+%\|' "$LOG" 2>/dev/null | tail -1 | tr -d '|' || echo "...")
            echo -n "GPU$GPU_ID:$PROGRESS "
        fi
    done
    echo ""
    
    sleep 30
done

# Wait for all and capture exit codes
FAILED=0
for PID in ${PIDS[@]}; do
    if ! wait $PID; then
        echo "Process $PID failed!"
        FAILED=1
    fi
done

if [ $FAILED -eq 1 ]; then
    echo "ERROR: Some processes failed. Check logs in $OUTPUT_BASE/*.log"
    exit 1
fi

echo ""
echo "============================================================"
echo "All processes completed! Merging results..."
echo "============================================================"

# Merge all traces into single file
MERGED="${OUTPUT_BASE}/traces.jsonl"
> $MERGED  # Clear/create file
for GPU_ID in 2 3 4 5 6 7; do
    if [ -f "${OUTPUT_BASE}/gpu${GPU_ID}/traces.jsonl" ]; then
        cat "${OUTPUT_BASE}/gpu${GPU_ID}/traces.jsonl" >> $MERGED
    fi
done

# Count and report
TOTAL_TRACES=$(wc -l < $MERGED)
echo "Merged $TOTAL_TRACES traces to: $MERGED"

# Calculate aggregate stats
python3 << PYTHON
import json

traces = []
with open("$MERGED") as f:
    for line in f:
        traces.append(json.loads(line))

correct = sum(1 for t in traces if t["correct"])
valid = sum(1 for t in traces if t.get("all_steps_valid", False))
total_steps = sum(t.get("total_step_count", 0) for t in traces)
valid_steps = sum(t.get("valid_step_count", 0) for t in traces)

print(f"\nTotal traces: {len(traces)}")
print(f"Correct answers: {correct} ({correct/len(traces)*100:.1f}%)")
print(f"Valid traces: {valid} ({valid/len(traces)*100:.1f}%)")
if total_steps:
    print(f"Step validity: {valid_steps}/{total_steps} ({valid_steps/total_steps*100:.1f}%)")
PYTHON

echo "============================================================"
