#!/bin/bash
# Evaluate baselines that DON'T require training
# Run SEQUENTIALLY to avoid vLLM memory contention issues
#
# 1. Base CoT (few-shot chain-of-thought)
# 2. Self-Consistency (k=8 samples, majority vote)
#
# Usage: ./scripts/eval_baselines.sh

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "BASELINE EVALUATIONS (Sequential - avoids memory issues)"
echo "============================================================"
echo ""
echo "Baselines:"
echo "  1. Base CoT - Few-shot prompting with Qwen3-8B"
echo "  2. Self-Consistency - Sample k=8, majority vote"
echo ""
echo "Datasets: PrOntoQA test (1594), FOLIO validation (203)"
echo "============================================================"

OUTPUT_DIR="outputs/eval/baselines"
mkdir -p $OUTPUT_DIR

############################################################
# Base CoT - PrOntoQA
############################################################
echo ""
echo "============================================================"
echo "[1/4] Base CoT on PrOntoQA"
echo "============================================================"

CUDA_VISIBLE_DEVICES=2 python scripts/eval_baselines_no_training.py \
    --baseline base_cot \
    --data prontoqa \
    --model Qwen/Qwen3-8B \
    --output-dir $OUTPUT_DIR \
    --gpu 0

############################################################
# Base CoT - FOLIO
############################################################
echo ""
echo "============================================================"
echo "[2/4] Base CoT on FOLIO"
echo "============================================================"

CUDA_VISIBLE_DEVICES=2 python scripts/eval_baselines_no_training.py \
    --baseline base_cot \
    --data folio \
    --model Qwen/Qwen3-8B \
    --output-dir $OUTPUT_DIR \
    --gpu 0

############################################################
# Self-Consistency - PrOntoQA
############################################################
echo ""
echo "============================================================"
echo "[3/4] Self-Consistency (k=8) on PrOntoQA"
echo "============================================================"

CUDA_VISIBLE_DEVICES=2 python scripts/eval_baselines_no_training.py \
    --baseline self_consistency \
    --data prontoqa \
    --model Qwen/Qwen3-8B \
    --output-dir $OUTPUT_DIR \
    --k 8 \
    --gpu 0

############################################################
# Self-Consistency - FOLIO
############################################################
echo ""
echo "============================================================"
echo "[4/4] Self-Consistency (k=8) on FOLIO"
echo "============================================================"

CUDA_VISIBLE_DEVICES=2 python scripts/eval_baselines_no_training.py \
    --baseline self_consistency \
    --data folio \
    --model Qwen/Qwen3-8B \
    --output-dir $OUTPUT_DIR \
    --k 8 \
    --gpu 0

############################################################
# FINAL SUMMARY
############################################################

echo ""
echo ""
echo "########################################################"
echo "# FINAL SUMMARY - ALL BASELINES"
echo "########################################################"

python3 << 'EOF'
import json
from pathlib import Path

output_dir = Path("outputs/eval/baselines")
results = {}

for name in ["prontoqa_base_cot", "folio_base_cot", "prontoqa_self_consistency", "folio_self_consistency"]:
    summary_path = output_dir / name / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results[name] = json.load(f)

print("")
print("Results:")
print("-" * 50)
for name, data in sorted(results.items()):
    acc = data.get('accuracy', 0)
    print(f"  {name:30} | Accuracy: {acc:6.1%}")

# Save combined
with open(output_dir / "combined_summary.json", "w") as f:
    json.dump(results, f, indent=2)
    
print("")
print(f"Combined summary: {output_dir}/combined_summary.json")
EOF
