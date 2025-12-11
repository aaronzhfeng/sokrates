# Baseline Evaluation Scripts

## Baselines WITHOUT Training (Quick to run)

These use the base Qwen3-8B model with different prompting strategies:

| Baseline | Script | Time | Description |
|----------|--------|------|-------------|
| Base CoT | `eval_baselines.sh` | ~30 min | Few-shot chain-of-thought prompting |
| Self-Consistency | `eval_baselines.sh` | ~2 hours | Sample k=8, majority vote |

```bash
# Run both baselines on both datasets
./scripts/eval_baselines.sh

# Or run specific baseline/dataset
python scripts/eval_baselines_no_training.py --baseline base_cot --data prontoqa
python scripts/eval_baselines_no_training.py --baseline self_consistency --data folio --k 8
```

---

## Baselines REQUIRING Training (More effort)

These require additional training runs:

### 1. Answer-only DPO
**Concept**: DPO trained only on (correct answer, wrong answer) pairs, without reasoning traces.

**To implement**:
```bash
# Would need to:
# 1. Generate traces with SFT model
# 2. Build preference pairs based ONLY on final answer correctness (ignore step validity)
# 3. Train DPO

# Not yet implemented - would need a modified train_dpo_from_traces.py
```

### 2. CoT-DPO (Unstructured)
**Concept**: DPO on raw chain-of-thought text (not optionized).

**To implement**:
```bash
# Would need to:
# 1. Train SFT on raw CoT (not optionized format)
# 2. Generate unstructured CoT traces
# 3. Build preference pairs based on answer correctness
# 4. Train DPO

# Not yet implemented - would need different data format
```

### 3. VeriCoT
**Concept**: From the VeriCoT paper - verify each CoT step and use for preference learning.

**Recommendation**: Pull numbers directly from their paper rather than re-implementing.

---

## Prior Methods (Pull from Papers)

For these, we recommend citing their reported numbers:

| Method | Paper | PrOntoQA Acc | Notes |
|--------|-------|--------------|-------|
| LoGiPT | Feng et al. 2024 | ~95% | Trains LLM to emulate solver |
| Logic-LM | Pan et al. 2023 | ~90% | Uses external solver |
| LINC | Olausson et al. 2023 | - | Focuses on FOL parsing |

---

## Quick Summary: What to Run

**Minimum for paper**:
1. `./scripts/eval_all_checkpoints.sh` - Our 7 models (~1 hour)
2. `./scripts/eval_baselines.sh` - Base CoT + Self-Consistency (~2.5 hours)

**Total**: ~3.5 hours of evaluation

**Results location**: `outputs/eval/`

