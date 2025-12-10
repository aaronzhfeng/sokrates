# SOKRATES Experimental Results

This folder contains metrics, summaries, and figures from our experiments.
Model checkpoints are NOT included (too large) but can be regenerated.

## Final Results: PrOntoQA

| Stage | Accuracy | Step Validity | Notes |
|-------|----------|---------------|-------|
| SFT (iter 0) | 93.3% | 11.3% | Initial supervised fine-tuning |
| DPO iter 1 | 96.8% | 44.7% | First solver-guided iteration |
| DPO iter 2 | 98.1% | 83.5% | Second iteration |
| DPO iter 3 | **98.2%** | 91.8% | Final model |

## Directory Structure

```
results/
├── sft/                    # SFT training metrics
│   ├── summary.json        # Final training summary
│   ├── training_history.json # Loss curves per step
│   ├── metrics.jsonl       # Logged metrics
│   └── config.json         # Training configuration
│
├── dpo/                    # DPO training metrics
│   ├── iter1/
│   │   ├── training_summary.json
│   │   └── dpo_config.json
│   ├── iter2/
│   └── iter3/
│
├── traces/                 # Trace generation summaries
│   ├── iter0/summary.json  # SFT model traces
│   ├── iter1/summary.json  # After DPO iter1
│   ├── iter2/summary.json  # After DPO iter2
│   └── quality_tests/      # Hyperparameter search
│       ├── quality_test_t0.0.json   # Temperature sweep
│       ├── quality_test_t0.3.json
│       ├── quality_test_t0.5.json
│       ├── quality_test_t0.7.json
│       ├── quality_test_t1.0.json
│       ├── quality_test_steps5.json  # Max steps sweep
│       ├── quality_test_steps15.json
│       └── quality_test_samples4.json # Samples per problem
│
├── eval/                   # Evaluation results
│   ├── dpo_iter1/summary.json
│   └── prontoqa_final/summary.json  # Final 98.2% accuracy
│
├── analysis/               # Analysis outputs
│   ├── hyperparameter_search.json
│   ├── hyperparameter_search.csv
│   ├── results.json
│   └── results_table.tex
│
└── figures/                # Training visualization
    ├── sft_loss.png
    ├── sft_lr.png
    ├── sft_grad_norm.png
    ├── sft_combined.png
    └── sft_training.png
```

## Key Findings

### Hyperparameter Search (Temperature)
| Temperature | Accuracy | Step Validity | Diversity |
|-------------|----------|---------------|-----------|
| 0.0 (greedy) | 90.0% | 40.5% | Low |
| 0.3 | 91.5% | 38.2% | Medium |
| 0.5 | 92.0% | 35.8% | Good |
| 0.7 | 90.5% | 32.1% | High |
| 1.0 | 85.0% | 28.4% | Very High |

**Best for DPO training**: T=0.5 (balances accuracy with diversity for preference pairs)

### Training Progression
- SFT teaches the Thought/Action format
- DPO iter1: Large improvement in step validity
- DPO iter2-3: Refinement, diminishing returns

## Regenerating Results

```bash
# Regenerate from scratch (requires ~6 B200 GPUs, ~3 hours)
./scripts/run_prontoqa_remaining.sh

# Or run full pipeline
./scripts/run_full_pipeline.sh
```

## Hardware Used
- 6× NVIDIA B200 (183GB VRAM each)
- GPUs 2-7 on mlsys-b200 cluster

