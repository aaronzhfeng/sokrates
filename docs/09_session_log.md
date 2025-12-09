# SOKRATES Development Session Log

This document tracks implementation progress, configuration changes, and lessons learned during development.

---

## Session: December 9, 2025

### Overview

Initial implementation and first SFT training run on NVIDIA H100 PCIe (80GB).

---

### 1. Data Preparation

#### Datasets Acquired

| Dataset | Source | Train | Test/Val |
|---------|--------|-------|----------|
| **FOLIO** | `yale-nlp/FOLIO` (HuggingFace, gated) | 1,001 | 203 |
| **PrOntoQA** | `jzfeng/LoGiPT-data` (HuggingFace) | 14,346 | 1,594 |

#### Data Pipeline Changes

1. **Updated `scripts/prepare_data.py`**:
   - Added support for `jzfeng/LoGiPT-data` (conversation format from LoGiPT NAACL'24 paper)
   - Implemented `parse_logipt_conversation()` to extract context, query, chain, and answer
   - Added automatic 90/10 train/test split for PrOntoQA
   - Removed synthetic data generation (production use only)
   - Added proper logging with timestamps

2. **Data Format**:
   - FOLIO: FOL premises with natural language, TRUE/FALSE/UNKNOWN labels
   - PrOntoQA: Ontology-based reasoning chains with implied facts

---

### 2. Model Selection

#### Final Choice: `Qwen/Qwen3-8B`

| Model | Status | Notes |
|-------|--------|-------|
| `meta-llama/Llama-3.1-8B-Instruct` | ❌ Gated | Requires HF login, license agreement |
| `Qwen/Qwen2.5-7B-Instruct` | ✅ Available | Good baseline |
| **`Qwen/Qwen3-8B`** | ✅ **Selected** | Enhanced reasoning capabilities, Apache 2.0 |

**Why Qwen3?**
- Improved "Thinking" mode for multi-step reasoning
- 36T training tokens (2x Qwen2.5)
- Better suited for logical reasoning tasks

---

### 3. Experiment Logging Infrastructure

#### Created `src/utils/logging.py`

| Component | Purpose |
|-----------|---------|
| `ExperimentLogger` | Creates timestamped experiment directories |
| `MetricsTracker` | Logs metrics to JSONL for plotting |
| `ExperimentConfig` | Serializable config for reproducibility |
| `setup_logging()` | Dual file + console logging with timestamps |

#### Output Structure

```
outputs/
└── sft/
    └── 20251209_222946/          # Timestamped run
        ├── run.log               # Full log with timestamps
        ├── config.json           # Experiment configuration
        ├── metrics.jsonl         # Metrics over time
        ├── training_history.json # HuggingFace trainer logs
        ├── summary.json          # Final summary
        └── final/                # Model checkpoint (LoRA adapter)
```

#### Key Features
- **Timestamped directories**: Runs never overwrite previous runs
- **`latest` symlink**: Points to most recent run
- **JSONL metrics**: Easy to parse for plotting

---

### 4. Training Configuration

#### SFT Settings (`configs/training.yaml`)

```yaml
sft:
  batch_size: 4              # Balanced for H100 80GB
  gradient_accumulation_steps: 8  # Effective batch = 32
  max_seq_length: 1024       # Reduced for memory efficiency
  num_epochs: 3
  learning_rate: 2.0e-5
  save_total_limit: 2        # Keep only 2 checkpoints (disk space)
```

#### LoRA Settings (`configs/model.yaml`)

```yaml
peft:
  enabled: true
  r: 64
  lora_alpha: 128
  lora_dropout: 0.05
  target_modules:
    - q_proj, k_proj, v_proj, o_proj
    - gate_proj, up_proj, down_proj
```

**Trainable Parameters**: 174.6M / 8.4B total (2.09%)

---

### 5. Issues Encountered & Solutions

#### Issue 1: `evaluation_strategy` Deprecated
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
```
**Solution**: Changed to `eval_strategy` in `src/training/sft.py`

#### Issue 2: LoRA Gradient Error
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```
**Solution**: 
- Moved gradient checkpointing AFTER LoRA application
- Used `gradient_checkpointing_kwargs={"use_reentrant": False}`

#### Issue 3: CUDA Out of Memory (batch_size=8)
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.27 GiB
```
**Solution**: Reduced batch_size to 4, increased gradient_accumulation_steps to 8

#### Issue 4: Model Config Not Loading
- Script had hardcoded fallback to Llama model
- **Solution**: Updated `load_config()` to also load `configs/model.yaml`

---

### 6. Hardware Utilization

| Metric | Initial (batch=2) | Optimized (batch=4) |
|--------|-------------------|---------------------|
| GPU Memory | ~30% (24GB) | ~50-60% (40-48GB) |
| GPU Utilization | ~19% | ~40-60% |
| Training Speed | ~11s/iter | ~5-6s/iter |

**Target**: 70-80% GPU memory utilization for optimal throughput.

---

### 7. File Sizes

| Category | Size |
|----------|------|
| Total data | 43 MB |
| Largest file | 23 MB (`prontoqa_train.jsonl`) |
| LoRA checkpoint | ~350-500 MB |
| Full project (excl. venv) | ~50 MB |

All files under GitHub's 100MB limit. Data files excluded via `.gitignore`.

---

### 8. Current Training Status

**Run ID**: `20251209_222946`

```
Model: Qwen/Qwen3-8B
Epochs: 3
Batch size: 4
Effective batch: 32
Training traces: 14,346
Estimated time: 1-2 hours
```

---

### 9. Next Steps

1. **Complete SFT training** - Monitor for completion
2. **Evaluate SFT model** - Run on PrOntoQA test set
3. **Run OaK-DPO loop** - 3 iterations with solver verification
4. **Ablation studies** - Test impact of each component
5. **Transfer to FOLIO** - Test generalization

---

### 10. Commands Reference

```bash
# Data preparation
python scripts/prepare_data.py --raw-dir data/raw --output-dir data/processed

# SFT training
python scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl

# OaK-DPO training (after SFT)
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/latest/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --val-data data/processed/prontoqa_test.jsonl \
    --iterations 3

# Evaluation
python scripts/evaluate.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa
```

---

## Lessons Learned

1. **Start with smaller batch sizes** - OOM errors are common; scale up gradually
2. **Check config loading** - Hardcoded defaults can override config files
3. **Use timestamped outputs** - Never overwrite previous experiment runs
4. **LoRA order matters** - Apply gradient checkpointing after LoRA setup
5. **Monitor GPU utilization** - Low utilization means room for larger batches

