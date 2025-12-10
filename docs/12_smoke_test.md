# SOKRATES Smoke Test Guide

Quick end-to-end tests to validate the pipeline before long runs.

---

## 1. Quick OaK-DPO Test (5-10 min)

Tests: trace generation → verification → preference pairs → DPO training → checkpoint saving

```bash
# Use GPU 1 (or any free GPU)
CUDA_VISIBLE_DEVICES=1 python scripts/run_oak_dpo.py \
    --config configs/toy_test.yaml \
    --sft-model outputs/sft/20251209_150417/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --val-data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --iterations 1 \
    --output-dir outputs/smoke_test/oak_$(date +%H%M%S)
```

**Expected output:**
- `outputs/smoke_test/oak_*/checkpoints/iter_0/` - DPO checkpoint
- `outputs/smoke_test/oak_*/metrics.jsonl` - Iteration metrics
- Console shows: trace generation, verification, DPO training progress

---

## 2. Evaluation Test (2-3 min)

Tests: model loading → trace generation → solver verification → metrics computation

```bash
# Test SFT model evaluation
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
    --model outputs/sft/20251209_150417/final \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --max-samples 10 \
    --output-dir outputs/smoke_test/eval \
    --dataset-name sft_quick
```

**Expected output:**
- `outputs/smoke_test/eval/sft_quick_metrics.json`
- `outputs/smoke_test/eval/sft_quick_report.txt`
- Console shows accuracy, step validity, trace validity

---

## 3. DPO Model Evaluation (after OaK completes)

```bash
# Test DPO checkpoint evaluation
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py \
    --model outputs/smoke_test/oak_*/checkpoints/iter_0 \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --max-samples 10 \
    --merge-adapter \
    --save-traces \
    --output-dir outputs/smoke_test/eval \
    --dataset-name dpo_quick
```

---

## 4. Figure Generation Test

```bash
# Test plotting (uses existing SFT data)
python scripts/generate_paper_figures.py \
    --sft-dir outputs/sft/20251209_150417 \
    --output-dir outputs/smoke_test/figures \
    --format png
```

---

## 5. Full Mini-Pipeline (15-20 min)

Run all stages sequentially:

```bash
#!/bin/bash
set -e
GPU=1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="outputs/smoke_test/$TIMESTAMP"

echo "=== SOKRATES Smoke Test ==="
echo "Output: $OUT"
echo "GPU: $GPU"

# 1. OaK-DPO (single iteration, 5 problems)
echo -e "\n[1/3] Running OaK-DPO..."
CUDA_VISIBLE_DEVICES=$GPU python scripts/run_oak_dpo.py \
    --config configs/toy_test.yaml \
    --sft-model outputs/sft/20251209_150417/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --val-data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --iterations 1 \
    --output-dir $OUT/oak

# 2. Evaluate SFT baseline
echo -e "\n[2/3] Evaluating SFT baseline..."
CUDA_VISIBLE_DEVICES=$GPU python scripts/evaluate.py \
    --model outputs/sft/20251209_150417/final \
    --data data/processed/prontoqa_test.jsonl \
    --max-samples 20 \
    --output-dir $OUT/eval \
    --dataset-name sft_baseline

# 3. Evaluate DPO model (if checkpoint exists)
if [ -d "$OUT/oak/checkpoints/iter_0" ]; then
    echo -e "\n[3/3] Evaluating DPO model..."
    CUDA_VISIBLE_DEVICES=$GPU python scripts/evaluate.py \
        --model $OUT/oak/checkpoints/iter_0 \
        --data data/processed/prontoqa_test.jsonl \
        --max-samples 20 \
        --merge-adapter \
        --output-dir $OUT/eval \
        --dataset-name dpo_iter0
fi

echo -e "\n=== Smoke Test Complete ==="
echo "Results: $OUT"
ls -la $OUT/
```

---

## Expected Results

| Stage | Time | Output Files |
|-------|------|--------------|
| OaK-DPO (20 problems × 2 samples) | ~3-5 min | `checkpoints/iter_0/`, `metrics.jsonl` |
| Validation (10 problems) | ~1-2 min | included in metrics |
| SFT Eval (20 samples) | ~2 min | `sft_baseline_metrics.json` |
| DPO Eval (20 samples) | ~2 min | `dpo_iter0_metrics.json` |

**Toy test config (`configs/toy_test.yaml`):**
- 20 training problems, 2 samples each = 40 traces
- 10 validation problems (quick eval)
- 1 iteration only
- Greedy decoding, no option head training

---

## Troubleshooting

**"No preference pairs"**: Normal for very small datasets - try `require_valid_winner: false`

**OOM on small GPU**: Reduce `batch_size` to 1, enable `gradient_checkpointing`

**Model mismatch error**: Check `adapter_config.json` points to correct base model

**Device mismatch**: Ensure `CUDA_VISIBLE_DEVICES` is set correctly

---

## Quick Checks

```bash
# Check GPU availability
nvidia-smi --query-gpu=index,memory.free --format=csv

# Check outputs exist
ls -la outputs/smoke_test/*/

# View metrics
cat outputs/smoke_test/*/metrics.jsonl
cat outputs/smoke_test/*/eval/*_report.txt
```

