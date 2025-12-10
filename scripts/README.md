# SOKRATES Scripts

This directory contains training, evaluation, and analysis scripts for the SOKRATES framework.

## Training Scripts

### `train_sft.py`
Supervised Fine-Tuning of the base LLM on optionized proof traces.

```bash
# 2-GPU training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 --mixed_precision=bf16 \
    scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl \
    --output-dir outputs/sft/$(date +%Y%m%d_%H%M%S) \
    --wandb

# Single GPU training
python scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl \
    --output-dir outputs/sft/test
```

### `run_oak_dpo.py`
OaK-DPO (Options and Knowledge - Direct Preference Optimization) training loop.

```bash
# 6-GPU training
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes=6 --mixed_precision=bf16 \
    scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/latest/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --val-data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --iterations 2 \
    --output-dir outputs/oak_dpo/$(date +%Y%m%d_%H%M%S) \
    --wandb
```

### `run_full_pipeline.sh`
Automated script to run the full SOKRATES pipeline (SFT → OaK-DPO → Evaluation).

```bash
CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_full_pipeline.sh
```

---

## Evaluation Scripts

### `evaluate.py`
Evaluate trained models on test data.

```bash
# Evaluate SFT baseline
python scripts/evaluate.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --output-dir outputs/evaluation/sft \
    --dataset-name sft_baseline

# Evaluate DPO model (with merged adapter for faster inference)
python scripts/evaluate.py \
    --model outputs/oak_dpo/latest/checkpoints/iter_1/model \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa \
    --output-dir outputs/evaluation/dpo \
    --merge-adapter \
    --save-traces

# Quick evaluation on subset
python scripts/evaluate.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_test.jsonl \
    --max-samples 100 \
    --dataset-name quick_test
```

**Arguments:**
- `--model`: Path to model checkpoint (PEFT adapter or full model)
- `--data`: Path to test data (JSONL format)
- `--dataset-type`: `prontoqa` or `folio` (determines solver)
- `--max-samples`: Limit number of samples for quick testing
- `--merge-adapter`: Merge LoRA weights for ~20% faster inference
- `--save-traces`: Save individual trace predictions to JSONL
- `--greedy`: Use greedy decoding (default: True)
- `--temperature`: Sampling temperature (0 = greedy)

---

## Analysis & Plotting Scripts

### `plot_training.py`
Generate publication-quality training curves and figures.

```bash
# Plot SFT training curves
python scripts/plot_training.py \
    --sft-dir outputs/sft/20251209_150417 \
    --output-dir outputs/figures \
    --format pdf

# Plot OaK-DPO iteration metrics
python scripts/plot_training.py \
    --oak-dir outputs/oak_dpo/latest \
    --output-dir outputs/figures \
    --format png

# Plot both
python scripts/plot_training.py \
    --sft-dir outputs/sft/latest \
    --oak-dir outputs/oak_dpo/latest \
    --output-dir outputs/figures \
    --format pdf \
    --no-show
```

**Generated Figures:**
- `sft_loss.{fmt}`: Training loss curve
- `sft_lr.{fmt}`: Learning rate schedule
- `sft_grad_norm.{fmt}`: Gradient norm over training
- `sft_combined.{fmt}`: 2x2 combined SFT metrics
- `oak_iterations.{fmt}`: Accuracy/validity across OaK iterations

### `analyze_results.py`
Comprehensive post-training analysis with LaTeX tables and JSON output.

```bash
# Full analysis
python scripts/analyze_results.py \
    --sft-dir outputs/sft/latest \
    --oak-dir outputs/oak_dpo/latest \
    --output-dir outputs/analysis \
    --format pdf \
    --no-show
```

**Generated Outputs:**
- `sft_training.{fmt}`: Combined SFT figure
- `oak_iterations.{fmt}`: OaK iteration figure
- `model_comparison.{fmt}`: SFT vs DPO comparison
- `results_table.tex`: LaTeX table for paper
- `results.json`: Machine-readable results

---

## Data Preparation

### `prepare_data.py`
Convert raw datasets to SOKRATES format with optionized traces.

```bash
# Process PrOntoQA
python scripts/prepare_data.py \
    --input data/raw/prontoqa \
    --output data/processed \
    --dataset prontoqa

# Process FOLIO
python scripts/prepare_data.py \
    --input data/raw/folio \
    --output data/processed \
    --dataset folio
```

---

## Example Workflow

```bash
# 1. Prepare data
python scripts/prepare_data.py --dataset prontoqa

# 2. Train SFT model
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 --mixed_precision=bf16 \
    scripts/train_sft.py --config configs/training.yaml --wandb

# 3. Run OaK-DPO loop
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --num_processes=6 --mixed_precision=bf16 \
    scripts/run_oak_dpo.py \
    --sft-model outputs/sft/latest/final \
    --iterations 2 --wandb

# 4. Evaluate final model
python scripts/evaluate.py \
    --model outputs/oak_dpo/latest/checkpoints/iter_1/model \
    --data data/processed/prontoqa_test.jsonl \
    --merge-adapter --save-traces

# 5. Generate paper figures
python scripts/analyze_results.py \
    --sft-dir outputs/sft/latest \
    --oak-dir outputs/oak_dpo/latest \
    --format pdf --no-show
```

---

## Notes

- All scripts support `--help` for full argument documentation
- Use `CUDA_VISIBLE_DEVICES` to control GPU assignment
- For multi-GPU, use `accelerate launch --num_processes=N`
- Always use `--mixed_precision=bf16` for B200/H100 GPUs
- Output directories automatically create timestamps
- Symlink `latest` always points to most recent run

