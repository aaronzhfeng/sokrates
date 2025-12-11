#!/bin/bash
# Save all experimental results for paper writing
# Run this AFTER completing both PrOntoQA and FOLIO pipelines

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "SAVING ALL RESULTS FOR PAPER"
echo "============================================================"

# Create results directory structure
mkdir -p results/{prontoqa,folio}/{sft,dpo,traces,eval}
mkdir -p results/figures

############################################################
# PrOntoQA Results
############################################################
echo ""
echo "[PrOntoQA] Copying results..."

# SFT metrics
if [ -f "outputs/sft/latest/trainer_state.json" ]; then
    cp outputs/sft/latest/trainer_state.json results/prontoqa/sft/
fi
if [ -f "outputs/sft/latest/training_args.bin" ]; then
    python3 -c "
import torch
args = torch.load('outputs/sft/latest/training_args.bin')
import json
with open('results/prontoqa/sft/training_args.json', 'w') as f:
    json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)
" 2>/dev/null || echo "  Could not convert training_args"
fi

# DPO metrics (all iterations)
for iter in 1 2 3; do
    if [ -d "outputs/dpo/iter${iter}" ]; then
        mkdir -p results/prontoqa/dpo/iter${iter}
        cp outputs/dpo/iter${iter}/*.json results/prontoqa/dpo/iter${iter}/ 2>/dev/null || true
        cp outputs/dpo/iter${iter}/trainer_state.json results/prontoqa/dpo/iter${iter}/ 2>/dev/null || true
    fi
done

# Trace summaries (all iterations)
for iter in 0 1 2; do
    if [ -d "outputs/traces/iter${iter}" ]; then
        mkdir -p results/prontoqa/traces/iter${iter}
        cp outputs/traces/iter${iter}/summary.json results/prontoqa/traces/iter${iter}/ 2>/dev/null || true
        # Count traces
        if [ -f "outputs/traces/iter${iter}/traces.jsonl" ]; then
            wc -l outputs/traces/iter${iter}/traces.jsonl > results/prontoqa/traces/iter${iter}/count.txt
        fi
    fi
done

# Evaluation results
if [ -d "outputs/eval/prontoqa_final" ]; then
    cp outputs/eval/prontoqa_final/*.json results/prontoqa/eval/ 2>/dev/null || true
fi

echo "[PrOntoQA] Done!"

############################################################
# FOLIO Results
############################################################
echo ""
echo "[FOLIO] Copying results..."

# DPO metrics (all iterations)
for iter in 1 2 3; do
    if [ -d "outputs/dpo/folio_iter${iter}" ]; then
        mkdir -p results/folio/dpo/iter${iter}
        cp outputs/dpo/folio_iter${iter}/*.json results/folio/dpo/iter${iter}/ 2>/dev/null || true
        cp outputs/dpo/folio_iter${iter}/trainer_state.json results/folio/dpo/iter${iter}/ 2>/dev/null || true
    fi
done

# Trace summaries
for iter in 0 1 2; do
    if [ -d "outputs/traces/folio_iter${iter}" ]; then
        mkdir -p results/folio/traces/iter${iter}
        cp outputs/traces/folio_iter${iter}/summary.json results/folio/traces/iter${iter}/ 2>/dev/null || true
        if [ -f "outputs/traces/folio_iter${iter}/traces.jsonl" ]; then
            wc -l outputs/traces/folio_iter${iter}/traces.jsonl > results/folio/traces/iter${iter}/count.txt
        fi
    fi
done

# Evaluation results
if [ -d "outputs/eval/folio_final" ]; then
    cp outputs/eval/folio_final/*.json results/folio/eval/ 2>/dev/null || true
fi

echo "[FOLIO] Done!"

############################################################
# Generate Summary Report
############################################################
echo ""
echo "[Summary] Generating results summary..."

python3 << 'EOF'
import json
import os
from pathlib import Path

summary = {
    "prontoqa": {},
    "folio": {}
}

# PrOntoQA results
for iter_num in range(4):
    iter_name = "sft" if iter_num == 0 else f"dpo_iter{iter_num}"
    
    # Try to find eval results
    eval_path = Path(f"outputs/eval/prontoqa_{'final' if iter_num == 3 else f'iter{iter_num}'}")
    if eval_path.exists():
        for f in eval_path.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    if "accuracy" in data or "step_validity" in data:
                        summary["prontoqa"][iter_name] = data
            except:
                pass

# Check trace summaries for validity stats
for iter_num in range(3):
    trace_path = Path(f"outputs/traces/iter{iter_num}/summary.json")
    if trace_path.exists():
        try:
            with open(trace_path) as f:
                data = json.load(f)
                iter_name = "sft_traces" if iter_num == 0 else f"dpo_iter{iter_num}_traces"
                summary["prontoqa"][iter_name] = {
                    "total_traces": data.get("total_traces", 0),
                    "valid_traces": data.get("valid_traces", 0),
                    "accuracy": data.get("accuracy", 0),
                    "step_validity": data.get("step_validity", 0)
                }
        except:
            pass

# FOLIO results
for iter_num in range(3):
    trace_path = Path(f"outputs/traces/folio_iter{iter_num}/summary.json")
    if trace_path.exists():
        try:
            with open(trace_path) as f:
                data = json.load(f)
                iter_name = f"dpo_iter{iter_num+1}_traces"
                summary["folio"][iter_name] = {
                    "total_traces": data.get("total_traces", 0),
                    "valid_traces": data.get("valid_traces", 0),
                    "accuracy": data.get("accuracy", 0),
                    "step_validity": data.get("step_validity", 0)
                }
        except:
            pass

# Save summary
with open("results/experiment_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Results summary saved to results/experiment_summary.json")
print(json.dumps(summary, indent=2))
EOF

############################################################
# Create final zip for download
############################################################
echo ""
echo "[Archive] Creating downloadable archive..."

# Create archive of results
cd /raid/zhf004/sokrates
tar -czvf sokrates_results.tar.gz results/ docs/ README.md paper/ 2>/dev/null || \
tar -czvf sokrates_results.tar.gz results/ docs/ README.md

echo ""
echo "============================================================"
echo "ALL RESULTS SAVED!"
echo "============================================================"
echo ""
echo "Files created:"
echo "  - results/                     (all metrics and summaries)"
echo "  - results/experiment_summary.json (combined summary)"
echo "  - sokrates_results.tar.gz      (downloadable archive)"
echo ""
echo "To download:"
echo "  scp user@server:/raid/zhf004/sokrates/sokrates_results.tar.gz ."
echo ""
echo "============================================================"

