# SOKRATES

**S**ymbolic **O**ption-**K**nowledge **R**easoning **A**lignment via **T**race **E**valuation with **S**olver

A neuro-symbolic approach to logical reasoning that instantiates Sutton's Options and Knowledge (OaK) architecture in a first-order logic micro-world.

## Overview

SOKRATES improves LLM logical reasoning by:

1. **Optionizing proofs** - Representing reasoning as sequences of discrete inference-rule options (`MODUS_PONENS`, `UNIV_INSTANTIATION`, etc.) rather than free-form text
2. **Solver verification** - Using FOL solvers to verify each reasoning step, providing ground-truth "knowledge"
3. **DPO alignment** - Distilling solver knowledge into the LLM via Direct Preference Optimization
4. **Iterative OaK loop** - Running generate→verify→train cycles to continuously improve

## Installation

```bash
# Clone the repository
git clone https://github.com/sokrates-project/sokrates.git
cd sokrates

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

```bash
# Download and process FOLIO and PrOntoQA datasets
python scripts/prepare_data.py --raw-dir data/raw --output-dir data/processed
```

### 2. Run Supervised Fine-Tuning

```bash
# Train on optionized traces to learn the Thought/Action format
python scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl \
    --output-dir outputs/sft
```

### 3. Run OaK-DPO Loop

```bash
# Run the iterative OaK training loop
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --iterations 3
```

### 4. Evaluate

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --model outputs/oak_loop/checkpoints/iter_2/model \
    --data data/processed/prontoqa_test.jsonl \
    --output-dir outputs/evaluation
```

## Project Structure

```
sokrates/
├── configs/                    # Configuration files
│   ├── model.yaml             # Model architecture config
│   ├── training.yaml          # Training hyperparameters
│   └── evaluation.yaml        # Evaluation settings
├── data/                       # Data directory
│   ├── raw/                   # Downloaded datasets
│   └── processed/             # Optionized training data
├── docs/                       # Documentation
│   ├── 00_index.md            # Documentation index
│   ├── 05_technical_spec.md   # Technical specification
│   └── 06_glossary.md         # Terminology glossary
├── scripts/                    # Run scripts
│   ├── prepare_data.py        # Data preparation
│   ├── train_sft.py           # SFT training
│   ├── run_oak_dpo.py         # OaK-DPO loop
│   └── evaluate.py            # Model evaluation
├── src/                        # Source code
│   ├── data/                  # Data structures and loading
│   │   ├── structures.py      # Core data classes
│   │   └── optionizer.py      # Proof optionization
│   ├── models/                # Model components
│   │   ├── option_head.py     # q̂_φ predictor
│   │   └── gvf_heads.py       # GVF auxiliary heads
│   ├── solvers/               # FOL verification
│   │   ├── base_solver.py     # Abstract solver interface
│   │   ├── folio_solver.py    # FOLIO/Z3 solver
│   │   └── prontoqa_solver.py # PrOntoQA solver
│   ├── training/              # Training pipelines
│   │   ├── sft.py             # Supervised fine-tuning
│   │   ├── dpo.py             # DPO training
│   │   └── oak_loop.py        # OaK iteration loop
│   ├── inference/             # Generation utilities
│   │   ├── constrained_decode.py
│   │   └── generate_trace.py
│   └── evaluation/            # Metrics and analysis
│       ├── metrics.py
│       └── calibration.py
├── tests/                      # Unit tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Key Concepts

### Options

Discrete inference-rule macros that form the reasoning vocabulary:

| Option | Description | Example |
|--------|-------------|---------|
| `MODUS_PONENS` | From P and P→Q, derive Q | "Socrates is a man" + "All men are mortal" → "Socrates is mortal" |
| `UNIV_INSTANTIATION` | From ∀x.P(x), derive P(c) | "All cats are mammals" → "Felix is a mammal" |
| `AND_ELIM` | From P∧Q, derive P or Q | "It's cold and rainy" → "It's cold" |
| `CONCLUDE` | Terminal step | Output TRUE/FALSE/UNKNOWN |

### Thought/Action Format

Each proof step consists of:
```
Thought: Since we know Socrates is a man (premise 0) and all men are mortal (premise 1), 
we can apply modus ponens.
Action: <Option type="MODUS_PONENS" args="[0, 1]" />
```

### Knowledge (q̂_φ)

The option success predictor learns to estimate P(step is valid | state, option), providing explicit "knowledge" about reasoning actions.

### OaK Loop

The iterative training cycle:
1. **Generate** - Sample optionized traces from current policy
2. **Verify** - Label each step with FOL solver
3. **Update q̂_φ** - Train option success predictor on solver labels
4. **Build preferences** - Create (valid, invalid) trace pairs
5. **DPO** - Update policy to prefer valid traces
6. **Repeat**

## Configuration

### Model Settings (`configs/model.yaml`)

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  torch_dtype: "bfloat16"

peft:
  enabled: true
  r: 64
  lora_alpha: 128
```

### Training Settings (`configs/training.yaml`)

```yaml
oak_loop:
  num_iterations: 3
  samples_per_problem: 8

dpo:
  beta: 0.1
  learning_rate: 5.0e-6
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× RTX 4090 (24GB) | 4× A40 (48GB each) |
| RAM | 64GB | 128GB |
| Storage | 100GB SSD | 500GB NVMe |

## Citation

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

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- FOLIO dataset: Yale LILY Lab
- PrOntoQA: Aman Madaan et al.
- Options and Knowledge framework: Rich Sutton and the Alberta Plan
