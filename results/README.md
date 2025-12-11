# SOKRATES Experimental Results

This folder contains metrics, summaries, and figures from our experiments.
Model checkpoints are NOT included (too large) but can be regenerated or downloaded from HuggingFace.

---

## Main Results: PrOntoQA

| Model | Accuracy | Step Validity | Trace Validity |
|-------|----------|---------------|----------------|
| Base Model (CoT) | 44.4% | N/A | N/A |
| Self-Consistency (k=8) | 53.8% | N/A | N/A |
| **SFT** | 94.2% | 27.3% | 2.1% |
| **DPO iter1** | 95.9% | 87.8% | 71.3% |
| **DPO iter2** | 97.6% | 98.5% | 92.0% |
| **DPO iter3** | **98.3%** | **98.7%** | **91.8%** |

## Main Results: FOLIO (Transfer)

| Model | Accuracy | Step Validity | Trace Validity |
|-------|----------|---------------|----------------|
| Base Model (CoT) | 42.9% | N/A | N/A |
| Self-Consistency (k=8) | 42.9% | N/A | N/A |
| PrOntoQA SFT → FOLIO | 45.3% | 46.5% | 9.9% |
| PrOntoQA DPO iter3 → FOLIO | 53.2% | 48.3% | 14.8% |
| **FOLIO DPO iter1** | 51.7% | 46.7% | 15.8% |
| **FOLIO DPO iter2** | 51.2% | 46.3% | 13.3% |
| **FOLIO DPO iter3** | **51.2%** | **47.2%** | **13.8%** |

## Ablation Results

| Model | Accuracy | Step Validity | Description |
|-------|----------|---------------|-------------|
| SOKRATES (DPO iter1) | 95.9% | 87.8% | Full approach |
| **w/o Solver (Answer-only DPO)** | 95.5% | ~31.6%* | Solver verification removed |

*Note: Step validity in ablation measures per-step correctness, not trace-level validity.

**Key Finding**: Without solver verification, accuracy remains high but step validity drops dramatically, showing that solver-in-the-loop is crucial for reasoning quality.

---

## Directory Structure

```
results/
├── README.md                    # This file
│
├── eval/                        # All evaluation results
│   ├── checkpoints/             # Per-checkpoint evaluations
│   │   ├── prontoqa_sft/
│   │   ├── prontoqa_dpo_iter1/
│   │   ├── prontoqa_dpo_iter2/
│   │   ├── prontoqa_dpo_iter3/
│   │   ├── folio_dpo_iter1/
│   │   ├── folio_dpo_iter2/
│   │   └── folio_dpo_iter3/
│   │
│   ├── baselines/               # No-training baselines
│   │   ├── prontoqa_base_cot/
│   │   ├── prontoqa_self_consistency/
│   │   ├── folio_base_cot/
│   │   └── folio_self_consistency/
│   │
│   ├── transfer/                # Cross-dataset transfer
│   │   ├── prontoqa_sft_to_folio/
│   │   ├── prontoqa_dpo_iter3_to_folio/
│   │   └── folio_dpo_iter3_to_prontoqa/
│   │
│   ├── dpo_iter1/               # Legacy eval (subset)
│   └── prontoqa_final/          # Legacy eval
│
├── ablations/                   # Ablation studies
│   └── wo_solver_answer_only_dpo/  # w/o solver verification
│
├── sft/                         # SFT training metrics
│   ├── summary.json
│   ├── training_history.json
│   ├── metrics.jsonl
│   └── config.json
│
├── dpo/                         # DPO training metrics
│   ├── iter1/
│   ├── iter2/
│   └── iter3/
│
├── traces/                      # Trace generation summaries
│   ├── iter0/                   # SFT model traces
│   ├── iter1/                   # After DPO iter1
│   ├── iter2/                   # After DPO iter2
│   └── quality_tests/           # Hyperparameter search
│
├── analysis/                    # Analysis outputs
│   ├── hyperparameter_search.json
│   ├── hyperparameter_search.csv
│   └── results_table.tex
│
└── figures/                     # Training visualizations
    ├── sft_loss.png
    ├── sft_combined.png
    └── ...
```

---

## Key Metrics Definitions

| Metric | Definition |
|--------|------------|
| **Accuracy** | % of problems with correct final answer |
| **Step Validity** | % of reasoning steps verified correct by solver |
| **Trace Validity** | % of traces with ALL steps valid |

---

## Hyperparameter Search (Temperature)

| Temperature | Accuracy | Step Validity | Best For |
|-------------|----------|---------------|----------|
| 0.0 (greedy) | 90.0% | 40.5% | Final evaluation |
| 0.3 | 91.5% | 38.2% | - |
| **0.5** | **92.0%** | **35.8%** | **DPO training** |
| 0.7 | 90.5% | 32.1% | - |
| 1.0 | 85.0% | 28.4% | - |

**Recommendation**: T=0.5 for trace generation (diversity), T=0.0 for evaluation (consistency)

---

## Hardware

- 6× NVIDIA B200 (183GB VRAM each)
- GPUs 2-7 on mlsys-b200 cluster
- Training time: ~3-4 hours total

---

## Regenerating Results

```bash
# Full PrOntoQA pipeline
./scripts/run_prontoqa_remaining.sh

# Full FOLIO pipeline  
./scripts/run_folio_full.sh

# Evaluate all checkpoints
./scripts/eval_all_checkpoints.sh

# Run baselines
./scripts/eval_baselines.sh

# Run ablations
./scripts/ablation_3_answer_only_dpo.sh
```

---

## HuggingFace Models

| Model | Link |
|-------|------|
| PrOntoQA SFT | `Moonlight556/sokrates-Qwen3-32B-prontoqa-sft` |
| PrOntoQA DPO iter1 | `Moonlight556/sokrates-Qwen3-32B-prontoqa-dpo-iter1` |
| PrOntoQA DPO iter2 | `Moonlight556/sokrates-Qwen3-32B-prontoqa-dpo-iter2` |
| PrOntoQA DPO iter3 | `Moonlight556/sokrates-Qwen3-32B-prontoqa-dpo-iter3` |
