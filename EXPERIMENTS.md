# SOKRATES Experiment Commands

Complete command reference for reproducing all experiments in the paper.

**Target Hardware:** 1× NVIDIA H100 PCIe (80GB VRAM)  
**Estimated Total Time:** 3-4 hours  
**Estimated Cost:** $7-10 @ $2.39/hr

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [Main Experiments](#3-main-experiments)
4. [Ablation Studies](#4-ablation-studies)
5. [Evaluation](#5-evaluation)
6. [Quick Reference](#6-quick-reference)

---

## 1. Environment Setup

### 1.1 Clone and Install

```bash
# Clone repository
git clone https://github.com/sokrates-project/sokrates.git
cd sokrates

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import src; print(f'SOKRATES v{src.__version__}')"
```

### 1.2 Verify GPU

```bash
# Check GPU
nvidia-smi

# Should show: NVIDIA H100 PCIe (80GB)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 1.3 Login to Services (Optional)

```bash
# HuggingFace (for model download)
huggingface-cli login

# Weights & Biases (for experiment tracking)
wandb login
```

---

## 2. Data Preparation

### 2.1 Download and Process All Datasets

```bash
python scripts/prepare_data.py \
    --raw-dir data/raw \
    --output-dir data/processed
```

**Expected Output:**
```
data/
├── raw/
│   ├── folio/
│   │   ├── train.json
│   │   └── validation.json
│   └── prontoqa/
│       ├── train.json
│       └── test.json
└── processed/
    ├── folio_train.jsonl
    ├── folio_validation.jsonl
    ├── prontoqa_train.jsonl
    └── prontoqa_test.jsonl
```

### 2.2 Verify Data

```bash
# Count examples
wc -l data/processed/*.jsonl

# Preview data format
head -1 data/processed/prontoqa_train.jsonl | python -m json.tool
```

---

## 3. Main Experiments

### 3.1 Experiment 1: Supervised Fine-Tuning (SFT)

**Purpose:** Train base model on optionized proof format  
**Time:** ~20-30 minutes

```bash
python scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl \
    --output-dir outputs/sft \
    --wandb
```

**Without WandB:**
```bash
python scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl \
    --output-dir outputs/sft
```

**Output:** `outputs/sft/final/` (model checkpoint)

---

### 3.2 Experiment 2: OaK-DPO Training Loop

**Purpose:** Main SOKRATES training with solver-guided DPO  
**Time:** ~2.5-3 hours

```bash
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --val-data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --iterations 3 \
    --wandb
```

**Without WandB:**
```bash
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --dataset-type prontoqa \
    --iterations 3
```

**Output:**
- `outputs/oak_loop/training_summary.json`
- `checkpoints/iter_0/`, `checkpoints/iter_1/`, `checkpoints/iter_2/`

---

### 3.3 Experiment 3: Transfer to FOLIO

**Purpose:** Test transfer from PrOntoQA to FOLIO  
**Time:** ~1-1.5 hours

```bash
# Fine-tune on FOLIO after PrOntoQA pretraining
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model checkpoints/iter_2/model \
    --train-data data/processed/folio_train.jsonl \
    --val-data data/processed/folio_validation.jsonl \
    --dataset-type folio \
    --iterations 2 \
    --output-dir outputs/oak_loop_folio \
    --wandb
```

---

## 4. Ablation Studies

### 4.1 Ablation A: No Constrained Decoding

```bash
# Modify config temporarily
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --iterations 3 \
    --output-dir outputs/ablation_no_constrained
```

Then evaluate with constrained decoding disabled.

### 4.2 Ablation B: No Option Head (q̂_φ)

Edit `configs/training.yaml`:
```yaml
oak_loop:
  train_option_head: false
```

Then run:
```bash
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --iterations 3 \
    --output-dir outputs/ablation_no_option_head
```

### 4.3 Ablation C: Raw CoT (No Optionization)

This requires a separate baseline script (standard CoT fine-tuning without options).

### 4.4 Ablation D: Single OaK Iteration

```bash
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --iterations 1 \
    --output-dir outputs/ablation_single_iter
```

### 4.5 Ablation E: More Samples (8 per problem)

```bash
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --iterations 3 \
    --samples 8 \
    --output-dir outputs/ablation_8_samples
```

---

## 5. Evaluation

### 5.1 Evaluate Final Model on PrOntoQA

```bash
python scripts/evaluate.py \
    --model checkpoints/iter_2/model \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --output-dir outputs/evaluation \
    --dataset-name prontoqa_test
```

### 5.2 Evaluate on FOLIO

```bash
python scripts/evaluate.py \
    --model checkpoints/iter_2/model \
    --data data/processed/folio_validation.jsonl \
    --dataset-type folio \
    --output-dir outputs/evaluation \
    --dataset-name folio_val
```

### 5.3 Evaluate SFT Baseline

```bash
python scripts/evaluate.py \
    --model outputs/sft/final \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --output-dir outputs/evaluation \
    --dataset-name sft_baseline
```

### 5.4 Evaluate Each OaK Iteration

```bash
for i in 0 1 2; do
    python scripts/evaluate.py \
        --model checkpoints/iter_${i}/model \
        --data data/processed/prontoqa_test.jsonl \
        --dataset-type prontoqa \
        --output-dir outputs/evaluation \
        --dataset-name iter_${i}
done
```

### 5.5 Run All Ablation Evaluations

```bash
# No constrained decoding
python scripts/evaluate.py \
    --model outputs/ablation_no_constrained/checkpoints/iter_2/model \
    --data data/processed/prontoqa_test.jsonl \
    --output-dir outputs/evaluation \
    --dataset-name ablation_no_constrained

# No option head
python scripts/evaluate.py \
    --model outputs/ablation_no_option_head/checkpoints/iter_2/model \
    --data data/processed/prontoqa_test.jsonl \
    --output-dir outputs/evaluation \
    --dataset-name ablation_no_option_head

# Single iteration
python scripts/evaluate.py \
    --model outputs/ablation_single_iter/checkpoints/iter_0/model \
    --data data/processed/prontoqa_test.jsonl \
    --output-dir outputs/evaluation \
    --dataset-name ablation_single_iter
```

---

## 6. Quick Reference

### Complete Pipeline (Copy-Paste Ready)

```bash
# === FULL EXPERIMENT PIPELINE ===
# Estimated time: 3-4 hours on H100 PCIe
# Estimated cost: $7-10 @ $2.39/hr

# 1. Setup (5 min)
cd sokrates
source venv/bin/activate

# 2. Data (10 min)
python scripts/prepare_data.py

# 3. SFT (25 min)
python scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl

# 4. OaK-DPO (2.5 hrs)
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --iterations 3

# 5. Evaluate (20 min)
python scripts/evaluate.py \
    --model checkpoints/iter_2/model \
    --data data/processed/prontoqa_test.jsonl \
    --output-dir outputs/evaluation

# 6. View results
cat outputs/evaluation/prontoqa_test_report.txt
```

### Key Output Files

| File | Description |
|------|-------------|
| `outputs/sft/final/` | SFT model checkpoint |
| `checkpoints/iter_2/model/` | Final OaK-DPO model |
| `checkpoints/iter_2/option_head.pt` | Trained q̂_φ |
| `outputs/oak_loop/training_summary.json` | Training metrics |
| `outputs/evaluation/*_metrics.json` | Evaluation results |
| `outputs/evaluation/*_report.txt` | Human-readable reports |

### Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size in configs/training.yaml
# sft.batch_size: 8 → 4
# dpo.batch_size: 4 → 2
```

**Slow Generation:**
```bash
# Reduce samples per problem
--samples 2  # Instead of 4
```

**WandB Issues:**
```bash
# Run without WandB
# Just remove --wandb flag from commands
```

---

## Expected Results

### Main Metrics (Table for Paper)

| Model | Accuracy | Step Validity | Trace Validity | Brier | ECE |
|-------|----------|---------------|----------------|-------|-----|
| Base CoT | ~60% | ~50% | ~30% | - | - |
| SFT | ~70% | ~65% | ~45% | - | - |
| OaK-DPO (iter 1) | ~75% | ~75% | ~55% | 0.20 | 0.15 |
| OaK-DPO (iter 2) | ~80% | ~80% | ~65% | 0.15 | 0.10 |
| OaK-DPO (iter 3) | ~82% | ~85% | ~70% | 0.12 | 0.08 |

*Note: Actual numbers will vary. These are estimates based on similar work.*

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{sokrates2026,
  title={SOKRATES: Distilling Symbolic Knowledge into Option-Level Reasoning 
         via Solver-Guided Preference Optimization},
  author={[Authors]},
  booktitle={AAAI-26 Bridge Workshop on Logical and Symbolic Reasoning 
             in Language Models},
  year={2026}
}
```

