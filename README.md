# SOKRATES

**S**ymbolic **O**ption-**K**nowledge **R**easoning **A**lignment via **T**race **E**valuation with **S**olver

A neuro-symbolic approach to logical reasoning that instantiates Sutton's Options and Knowledge (OaK) architecture in a first-order logic micro-world.

## Overview

SOKRATES improves LLM logical reasoning by:

1. **Optionizing proofs** - Representing reasoning as sequences of discrete inference-rule options (`MODUS_PONENS`, `UNIV_INSTANTIATION`, etc.) rather than free-form text
2. **Solver verification** - Using FOL solvers to verify each reasoning step, providing ground-truth "knowledge"
3. **DPO alignment** - Distilling solver knowledge into the LLM via Direct Preference Optimization
4. **Iterative OaK loop** - Running generate→verify→train cycles to continuously improve

## 🏆 Results

| Model | PrOntoQA Accuracy | Step Validity |
|-------|-------------------|---------------|
| Base LLM (Qwen3-8B) | ~85% | - |
| + SFT | 93.3% | 11.3% |
| + DPO iter1 | 96.8% | 44.7% |
| + DPO iter2 | 98.1% | 83.5% |
| **+ DPO iter3** | **98.2%** | **91.8%** |

## 🤗 Pretrained Models

All models are available on HuggingFace:

| Model | Accuracy | Download |
|-------|----------|----------|
| SFT (Optionized) | 93.3% | [`sokrates-qwen3-8b-prontoqa-sft-optionized`](https://huggingface.co/Moonlight556/sokrates-qwen3-8b-prontoqa-sft-optionized) |
| DPO Iteration 1 | 96.8% | [`sokrates-qwen3-8b-prontoqa-oak-dpo-iter1`](https://huggingface.co/Moonlight556/sokrates-qwen3-8b-prontoqa-oak-dpo-iter1) |
| DPO Iteration 2 | 98.1% | [`sokrates-qwen3-8b-prontoqa-oak-dpo-iter2`](https://huggingface.co/Moonlight556/sokrates-qwen3-8b-prontoqa-oak-dpo-iter2) |
| **DPO Iteration 3** | **98.2%** | [`sokrates-qwen3-8b-prontoqa-oak-dpo-iter3`](https://huggingface.co/Moonlight556/sokrates-qwen3-8b-prontoqa-oak-dpo-iter3) |

### Load Pretrained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the best model (98.2% accuracy)
model = AutoModelForCausalLM.from_pretrained(
    "Moonlight556/sokrates-qwen3-8b-prontoqa-oak-dpo-iter3",
    torch_dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "Moonlight556/sokrates-qwen3-8b-prontoqa-oak-dpo-iter3"
)
```

## 📊 Dataset

Training data and generated traces are available on HuggingFace:

**Dataset**: [`Moonlight556/sokrates-prontoqa-data`](https://huggingface.co/datasets/Moonlight556/sokrates-prontoqa-data)

| File | Description | Records |
|------|-------------|---------|
| `processed/prontoqa_train.jsonl` | SFT training data with optionized traces | 14,346 |
| `processed/prontoqa_test.jsonl` | Test set | 1,594 |
| `processed/folio_train.jsonl` | FOLIO training data | 1,001 |
| `processed/folio_validation.jsonl` | FOLIO validation | 203 |
| `traces/iter0_traces.jsonl` | Traces from SFT model | 28,692 |
| `traces/iter1_traces.jsonl` | Traces from DPO iter1 | 28,692 |
| `traces/iter2_traces.jsonl` | Traces from DPO iter2 | 28,692 |
| `eval/prontoqa_final/traces.jsonl` | Final evaluation traces | 1,594 |

### Load Dataset

```python
from datasets import load_dataset

# Load training data
train_data = load_dataset(
    "Moonlight556/sokrates-prontoqa-data", 
    data_files="processed/prontoqa_train.jsonl",
    split="train"
)

# Load DPO traces
traces = load_dataset(
    "Moonlight556/sokrates-prontoqa-data",
    data_files="traces/iter0_traces.jsonl", 
    split="train"
)
```

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

## Methodology

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SOKRATES Pipeline                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  Raw     │───▶│ Optionize    │───▶│  SFT Training                │  │
│  │  Dataset │    │ Proofs       │    │  (Learn Thought/Action)      │  │
│  └──────────┘    └──────────────┘    └──────────────────────────────┘  │
│                                                    │                    │
│                                                    ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     OaK-DPO Loop (×3 iterations)                 │  │
│  │  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌─────────┐  │  │
│  │  │ Generate   │──▶│ Verify w/  │──▶│ Build DPO  │──▶│ Train   │  │  │
│  │  │ Traces     │   │ Solver     │   │ Pairs      │   │ DPO     │  │  │
│  │  └────────────┘   └────────────┘   └────────────┘   └─────────┘  │  │
│  │        ▲                                                  │      │  │
│  │        └──────────────────────────────────────────────────┘      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                    │                    │
│                                                    ▼                    │
│                                           ┌──────────────┐             │
│                                           │  Evaluation  │             │
│                                           └──────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Optionization

We transform free-form natural language proofs into structured **Thought/Action** sequences:

**Original proof (PrOntoQA):**
> "Wren is a jompus. Each jompus is nervous. Therefore, Wren is nervous."

**Optionized format:**
```
Premises:
  [0] Wren is a jompus.
  [1] Each jompus is nervous.

Conclusion to evaluate: Wren is nervous.

Determine if the conclusion is TRUE, FALSE, or UNKNOWN.

Reasoning:
Thought: Since Wren is a jompus (premise 0) and each jompus is nervous (premise 1), 
we can conclude that Wren is nervous.
Action: <Option type="MODUS_PONENS" args="[0, 1]" />
Thought: This matches our conclusion exactly, so it must be TRUE.
Action: <Option type="CONCLUDE" args="[0]" />
```

### Step 2: SFT Training

The base LLM is fine-tuned on optionized traces to learn:
- The structured Thought/Action output format
- How to reference premises by index
- When to apply different inference rules

**SFT Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen3-8B |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Learning Rate | 2e-5 |
| Batch Size | 4 (gradient accumulation: 4) |
| Epochs | 3 |
| Max Length | 2048 |

### Step 3: OaK-DPO Loop

Each iteration:

1. **Generate Traces**: Sample 2 traces per problem with temperature=0.5
2. **Verify with Solver**: Check each reasoning step for logical validity
3. **Build Preference Pairs**: Pair valid traces (chosen) with invalid traces (rejected)
4. **Train DPO**: Update model to prefer valid reasoning

**Trace Generation Hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature | 0.5 | Balance diversity vs quality |
| Max Steps | 15 | Maximum reasoning steps |
| Samples per Problem | 2 | Traces generated per problem |
| vLLM Tensor Parallel | 1 | Per-GPU parallelism |
| Data Parallel GPUs | 6 | Total parallelism |

**DPO Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Beta | 0.1 |
| Learning Rate | 5e-7 |
| Batch Size | 4 |
| Epochs | 1 |
| Max Length | 2048 |

### Step 4: Solver Verification

The solver validates each reasoning step by checking:
- **Premise indices**: Are the referenced premises valid?
- **Rule application**: Is the inference rule correctly applied?
- **Derivation**: Does the conclusion follow from the premises?

For PrOntoQA, we use a pattern-matching solver that understands the syllogistic structure.
For FOLIO, we use Z3 SMT solver for first-order logic verification.

## Key Concepts

### Options

Discrete inference-rule macros that form the reasoning vocabulary:

| Option | Description | Args | Example |
|--------|-------------|------|---------|
| `MODUS_PONENS` | From P and P→Q, derive Q | `[p_idx, rule_idx]` | "Socrates is a man" + "All men are mortal" → "Socrates is mortal" |
| `MODUS_TOLLENS` | From ¬Q and P→Q, derive ¬P | `[neg_q_idx, rule_idx]` | "Not mortal" + "Men are mortal" → "Not a man" |
| `UNIV_INSTANTIATION` | From ∀x.P(x), derive P(c) | `[forall_idx]` | "All cats are mammals" → "Felix is a mammal" |
| `AND_ELIM` | From P∧Q, derive P or Q | `[conj_idx]` | "It's cold and rainy" → "It's cold" |
| `CONCLUDE` | Terminal step | `[0/1/2]` | 0=TRUE, 1=FALSE, 2=UNKNOWN |

### Thought/Action Format

Each proof step consists of two parts:

```
Thought: <natural language explanation of reasoning>
Action: <Option type="RULE_NAME" args="[premise_indices]" />
```

**Complete Example:**
```
Premises:
  [0] Alex is a tumpus.
  [1] Every tumpus is small.
  [2] Every tumpus is a wumpus.
  [3] Every wumpus is floral.

Conclusion to evaluate: Alex is floral.

Reasoning:
Thought: Since Alex is a tumpus (premise 0) and every tumpus is a wumpus (premise 2), 
we can derive that Alex is a wumpus.
Action: <Option type="MODUS_PONENS" args="[0, 2]" />
Thought: Now that we know Alex is a wumpus (derived) and every wumpus is floral (premise 3), 
we can conclude Alex is floral.
Action: <Option type="MODUS_PONENS" args="[4, 3]" />
Thought: This matches our target conclusion, so the answer is TRUE.
Action: <Option type="CONCLUDE" args="[0]" />
```

### DPO Preference Pairs

We construct preference pairs for DPO training:

| Criteria | Chosen (Preferred) | Rejected |
|----------|-------------------|----------|
| Validity | Solver-verified valid | Contains invalid steps |
| Correctness | Reaches correct answer | Wrong final answer |
| Efficiency | Fewer steps | More steps (same answer) |

**Example Pair:**

*Chosen* (valid trace):
```
Thought: Using premise 0 and 1...
Action: <Option type="MODUS_PONENS" args="[0, 1]" />
Action: <Option type="CONCLUDE" args="[0]" />
```

*Rejected* (invalid trace):
```
Thought: Using premise 0 and 3...
Action: <Option type="MODUS_PONENS" args="[0, 3]" />  # Invalid: premises don't chain
Action: <Option type="CONCLUDE" args="[1]" />  # Wrong answer
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

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of problems with correct final answer (TRUE/FALSE/UNKNOWN) |
| **Step Validity** | % of reasoning steps that are logically valid per solver |
| **Valid Trace %** | % of complete traces with all steps valid |
| **Avg Steps** | Average number of reasoning steps per problem |

### Results Breakdown

| Iteration | Accuracy | Step Validity | Valid Traces | Avg Steps |
|-----------|----------|---------------|--------------|-----------|
| SFT | 93.3% | 11.3% | 3.2% | 2.8 |
| DPO iter1 | 96.8% | 44.7% | 28.5% | 3.1 |
| DPO iter2 | 98.1% | 83.5% | 71.2% | 3.0 |
| DPO iter3 | 98.2% | 91.8% | 85.4% | 2.9 |

**Key Insight**: While accuracy plateaus after iter2, step validity continues improving significantly, indicating the model learns more rigorous reasoning.

## Configuration

### Model Settings

```yaml
model:
  name: "Qwen/Qwen3-8B"           # Base model
  torch_dtype: "bfloat16"          # Mixed precision

peft:
  enabled: true
  r: 64                            # LoRA rank
  lora_alpha: 128                  # LoRA scaling
  target_modules:                  # Modules to adapt
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### SFT Training Settings

```yaml
sft:
  learning_rate: 2.0e-5
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4   # Effective batch: 16
  max_seq_length: 2048
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  gradient_checkpointing: true
```

### DPO Training Settings

```yaml
dpo:
  beta: 0.1                        # KL penalty weight
  learning_rate: 5.0e-7            # Lower than SFT
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  max_length: 2048
  max_prompt_length: 1024
```

### Trace Generation Settings

```yaml
generation:
  temperature: 0.5                 # Sampling temperature
  max_steps: 15                    # Max reasoning steps
  samples_per_problem: 2           # Traces per problem
  tensor_parallel_size: 1          # GPUs per model
  data_parallel_gpus: 6            # Parallel processes
```

## Hardware Requirements

| Component | Minimum | Recommended | Used in Paper |
|-----------|---------|-------------|---------------|
| GPU | 1× RTX 4090 (24GB) | 4× A100 (80GB each) | 6× B200 (183GB each) |
| RAM | 64GB | 128GB | 256GB |
| Storage | 100GB SSD | 500GB NVMe | 1TB NVMe |

**Training Time** (on 6× B200):
- SFT: ~50 minutes
- DPO per iteration: ~20 minutes
- Full pipeline (SFT + 3 DPO): ~2 hours

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
