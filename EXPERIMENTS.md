# SOKRATES Experiment Commands

Complete command reference for reproducing all experiments in the paper.

---

## ⚡ Current Status (Dec 10, 2025)

| Phase | Status | Notes |
|-------|--------|-------|
| **SFT Training** | ✅ Complete | `outputs/sft/latest/final` |
| **OaK-DPO Loop** | 🚀 Ready | Two-stage pipeline |
| **Evaluation** | ⏳ Pending | |

### Time-Optimized Settings (6-hour deadline)

Due to time constraints, we made these **principled compromises**:

| Parameter | Full Run | Optimized | Rationale |
|-----------|----------|-----------|-----------|
| Problems | 14,346 | **1,500** | 10% sample, statistically valid |
| Samples/problem | 8 | **2** | Min for preference pairs |
| Iterations | 3 | **2** | Still shows improvement curve |
| Max steps | 15 | **6** | PrOntoQA avg is 3-5 steps |
| Sampling | Stochastic | **Greedy** | Faster, deterministic |
| Option head | Train | **Skip** | Not critical for main result |

**Note:** These settings are sufficient for a proof-of-concept. For full paper results, use original settings with more compute time.

---

**Target Hardware:** 6× NVIDIA B200 (GPUs 2-7, 183GB VRAM each)  
**Estimated Total Time:** ~2-3 hours (optimized) | 6-8 hours (full)

| Experiment | Full Time | Optimized Time |
|------------|-----------|----------------|
| SFT Training | ~15 min | ✅ Done |
| OaK-DPO Loop | ~4-5 hours | ~1.5-2 hours |
| Evaluation | ~30 min | ~15 min |

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

# Should show: 2× NVIDIA B200 (183GB each)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, GPU 0: {torch.cuda.get_device_name(0)}')"
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

### 3.1 Experiment 1: Supervised Fine-Tuning (SFT) ✅ COMPLETE

**Purpose:** Train base model on optionized proof format  
**Status:** ✅ Complete - Model at `outputs/sft/latest/final`  
**Time taken:** ~15 minutes (2× B200)

```bash
# Multi-GPU training with accelerate (2× B200)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 --mixed_precision=bf16 \
    scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl \
    --output-dir outputs/sft \
    --wandb
```

**Without WandB:**
```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 --mixed_precision=bf16 \
    scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl \
    --output-dir outputs/sft
```

**Single GPU (if needed):**
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl \
    --output-dir outputs/sft
```

**Output:** `outputs/sft/latest/final/` (model checkpoint with LoRA adapter)

---

### 3.2 Experiment 2: OaK-DPO Training Loop (Two-Stage Pipeline)

**Purpose:** Main SOKRATES training with solver-guided DPO  
**Time:** ~1.5-2 hours (6× B200, optimized settings)

The OaK-DPO loop consists of two stages per iteration:
1. **Stage 1 (Trace Generation):** Generate optionized traces and verify with solver
2. **Stage 2 (DPO Training):** Build preference pairs and train DPO

#### Option A: Automated Loop (Recommended)

```bash
# Full OaK loop with 2 iterations, 1500 problems, on GPUs 2-7
./scripts/run_oak_loop.sh 2 1500 "2,3,4,5,6,7"
```

Output: `outputs/oak_loop_{timestamp}/`

#### Option B: Manual Stage-by-Stage (For Debugging)

**Stage 1: Generate Traces**
```bash
# Single GPU (simpler, ~30 min for 1500 problems)
CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/iter0 \
    --num-problems 1500 \
    --samples-per-problem 2 \
    --max-steps 6 \
    --temperature 0.0
```

```bash
# Multi-GPU parallel (faster, ~10 min for 1500 problems)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/generate_traces.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/iter0 \
    --num-problems 1500 \
    --samples-per-problem 2 \
    --num-gpus 6
```

**Stage 2: Train DPO from Traces**
```bash
# Multi-GPU DPO training (~20 min)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter0/traces.jsonl \
    --model outputs/sft/latest/final \
    --output outputs/dpo/iter0 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --beta 0.1
```

```bash
# Single GPU DPO (slower, ~45 min)
CUDA_VISIBLE_DEVICES=2 python scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter0/traces.jsonl \
    --model outputs/sft/latest/final \
    --output outputs/dpo/iter0 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 8
```

**Iteration 2 (use DPO model from iter0):**
```bash
# Generate traces with DPO model
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/generate_traces.py \
    --model outputs/dpo/iter0/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/iter1 \
    --num-problems 1500 \
    --samples-per-problem 2 \
    --num-gpus 6

# Train DPO
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter1/traces.jsonl \
    --model outputs/dpo/iter0/final \
    --output outputs/dpo/iter1
```

**Output Structure:**
```
outputs/
├── traces/
│   ├── iter0/
│   │   ├── traces.jsonl       # Generated traces with solver verification
│   │   └── summary.json       # Statistics
│   └── iter1/
├── dpo/
│   ├── iter0/
│   │   ├── final/             # DPO model checkpoint
│   │   └── preference_pairs.jsonl
│   └── iter1/
└── oak_loop_latest -> oak_loop_{timestamp}/  # If using automated loop
```

---

### 3.3 Experiment 3: Transfer to FOLIO

**Purpose:** Test transfer from PrOntoQA to FOLIO  
**Time:** ~2-3 hours (2× B200)

```bash
# Fine-tune on FOLIO after PrOntoQA pretraining (2× B200)
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 \
    scripts/run_oak_dpo.py \
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
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 \
    scripts/run_oak_dpo.py \
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
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 \
    scripts/run_oak_dpo.py \
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
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 \
    scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --iterations 1 \
    --output-dir outputs/ablation_single_iter
```

### 4.5 Ablation E: More Samples (16 per problem)

```bash
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 \
    scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --iterations 3 \
    --samples 16 \
    --output-dir outputs/ablation_16_samples
```

---

## 5. Evaluation

### 5.1 Evaluate Final Model on PrOntoQA

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --model outputs/dpo/iter1/final \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --output-dir outputs/evaluation \
    --dataset-name final_model
```

### 5.2 Evaluate on FOLIO

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --model outputs/dpo/iter1/final \
    --data data/processed/folio_validation.jsonl \
    --dataset-type folio \
    --output-dir outputs/evaluation \
    --dataset-name folio_val
```

### 5.3 Evaluate SFT Baseline

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --output-dir outputs/evaluation \
    --dataset-name sft_baseline
```

### 5.4 Evaluate Each OaK Iteration

```bash
# SFT baseline
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --output-dir outputs/evaluation \
    --dataset-name iter_0_sft

# After iteration 0
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --model outputs/dpo/iter0/final \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --output-dir outputs/evaluation \
    --dataset-name iter_1_dpo

# After iteration 1 (final)
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --model outputs/dpo/iter1/final \
        --data data/processed/prontoqa_test.jsonl \
        --dataset-type prontoqa \
        --output-dir outputs/evaluation \
    --dataset-name iter_2_dpo
```

### 5.5 Run All Ablation Evaluations

```bash
# No constrained decoding (use traces generated without grammar constraints)
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --model outputs/ablation_no_constrained/dpo/iter1/final \
    --data data/processed/prontoqa_test.jsonl \
    --output-dir outputs/evaluation \
    --dataset-name ablation_no_constrained

# No option head (current optimized run already skips this)
# Main results serve as this ablation

# Single iteration
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --model outputs/dpo/iter0/final \
    --data data/processed/prontoqa_test.jsonl \
    --output-dir outputs/evaluation \
    --dataset-name ablation_single_iter
```

---

## 6. Quick Reference

### Complete Pipeline (Copy-Paste Ready)

```bash
# === FULL EXPERIMENT PIPELINE ===
# Estimated time: 2-3 hours (optimized) on 6× B200
# Hardware: 6× NVIDIA B200 (GPUs 2-7, 183GB VRAM each)

# 1. Setup (5 min)
cd /raid/zhf004/sokrates
source venv/bin/activate

# 2. Data (10 min) - if not already done
python scripts/prepare_data.py

# 3. SFT - ALREADY COMPLETE ✅
# Model at: outputs/sft/latest/final

# 4. OaK-DPO Two-Stage Loop (~1.5-2 hrs)
# Option A: Automated (recommended)
./scripts/run_oak_loop.sh 2 1500 "2,3,4,5,6,7"

# Option B: Manual iteration by iteration
# --- Iteration 0 ---
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/generate_traces.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/iter0 \
    --num-problems 1500 \
    --samples-per-problem 2 \
    --num-gpus 6

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter0/traces.jsonl \
    --model outputs/sft/latest/final \
    --output outputs/dpo/iter0

# --- Iteration 1 ---
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/generate_traces.py \
    --model outputs/dpo/iter0/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/iter1 \
    --num-problems 1500 \
    --samples-per-problem 2 \
    --num-gpus 6

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter1/traces.jsonl \
    --model outputs/dpo/iter0/final \
    --output outputs/dpo/iter1

# 5. Evaluate (~15 min)
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py \
    --model outputs/dpo/iter1/final \
    --data data/processed/prontoqa_test.jsonl \
    --output-dir outputs/evaluation \
    --dataset-name final_model

# 6. View results
cat outputs/evaluation/final_model_report.txt
```

### Quick Start (Optimized Run)

```bash
# One-liner to run everything (after SFT is complete)
cd /raid/zhf004/sokrates && source venv/bin/activate && ./scripts/run_oak_loop.sh 2 1500 "2,3,4,5,6,7"
```

### Key Output Files

| File | Description |
|------|-------------|
| `outputs/sft/latest/final/` | SFT model checkpoint ✅ |
| `outputs/traces/iter{N}/traces.jsonl` | Generated traces with solver verification |
| `outputs/traces/iter{N}/summary.json` | Trace generation statistics |
| `outputs/dpo/iter{N}/final/` | DPO model checkpoint |
| `outputs/dpo/iter{N}/preference_pairs.jsonl` | Preference pairs used for DPO |
| `outputs/dpo/iter1/final/` | **Final model** (after 2 iterations) |
| `outputs/evaluation/*_metrics.json` | Evaluation results |
| `outputs/evaluation/*_report.txt` | Human-readable reports |

### Troubleshooting

**Out of Memory (unlikely with B200):**
```bash
# Reduce batch size in configs/training.yaml
# sft.batch_size: 16 → 8
# dpo.batch_size: 16 → 8
```

**GPU Selection:**
```bash
# Use specific GPUs (e.g., GPUs 1 and 2, avoiding GPU 0 if in use)
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 ...

# Single GPU
CUDA_VISIBLE_DEVICES=1 python ...
```

**Slow Generation:**
```bash
# Reduce samples per problem
--samples 4  # Instead of 8
```

**WandB Issues:**
```bash
# Run without WandB
# Just remove --wandb flag from commands
```

**Multi-GPU Setup:**
```bash
# Initialize accelerate config (run once)
accelerate config
# Select: multi-GPU, 2 GPUs, bf16
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

