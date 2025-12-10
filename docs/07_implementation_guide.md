# SOKRATES Implementation Guide

This document provides a comprehensive walkthrough of the SOKRATES codebase, explaining the architecture, key components, and how they work together.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Structures](#2-data-structures)
3. [Option Vocabulary](#3-option-vocabulary)
4. [Solvers](#4-solvers)
5. [Model Components](#5-model-components)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference](#7-inference)
8. [Evaluation](#8-evaluation)
9. [Configuration](#9-configuration)
10. [Usage Examples](#10-usage-examples)

---

## 1. Architecture Overview

SOKRATES implements the **Options and Knowledge (OaK)** framework for logical reasoning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SOKRATES System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                                           │
│   │   Problem   │  LogicalState: premises, conclusion, FOL formulas         │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐  ┌─────────────┐                                         │
│   │    LLM +    │  │   Option    │  q̂_φ(s,ω): predicts step validity       │
│   │   LoRA      │──│    Head     │                                         │
│   └──────┬──────┘  └─────────────┘                                         │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ Constrained │  Grammar ensures valid: <Option type="X" args="[...]" /> │
│   │  Decoding   │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │OptionizedTrace│ Sequence of ProofSteps (Thought + Action)              │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │ FOL Solver  │  Verifies each step: VALID / INVALID                     │
│   └──────┬──────┘                                                           │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │    DPO      │  Preference pairs: valid traces > invalid traces         │
│   │  Training   │                                                           │
│   └─────────────┘                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Organization

```
src/
├── data/           # Data structures and processing
│   ├── structures.py    # Core classes: OptionType, FOLFormula, ProofStep, etc.
│   └── optionizer.py    # Convert raw proofs to optionized format
├── models/         # Neural network components
│   ├── option_head.py   # q̂_φ predictor (Option Success Head)
│   └── gvf_heads.py     # GVF auxiliary heads
├── solvers/        # FOL verification
│   ├── base_solver.py   # Abstract solver interface
│   ├── folio_solver.py  # Z3-based solver for FOLIO
│   └── prontoqa_solver.py  # Ontology-based solver for PrOntoQA
├── training/       # Training pipelines
│   ├── sft.py          # Supervised Fine-Tuning
│   ├── dpo.py          # Direct Preference Optimization
│   └── oak_loop.py     # OaK training loop
├── inference/      # Generation utilities
│   ├── constrained_decode.py  # Grammar constraints
│   └── generate_trace.py      # Trace generation
└── evaluation/     # Metrics and analysis
    ├── metrics.py       # Accuracy, validity, Brier, ECE
    └── calibration.py   # Calibration analysis
```

---

## 2. Data Structures

### 2.1 Location: `src/data/structures.py`

This module defines all core data classes used throughout SOKRATES.

### OptionType (Enum)

Defines the vocabulary of inference rule options:

```python
class OptionType(Enum):
    MODUS_PONENS = "MP"           # From P and P→Q, derive Q
    MODUS_TOLLENS = "MT"          # From ¬Q and P→Q, derive ¬P
    UNIV_INSTANTIATION = "UI"     # From ∀x.P(x), derive P(c)
    AND_INTRO = "AI"              # From P and Q, derive P∧Q
    AND_ELIM = "AE"               # From P∧Q, derive P or Q
    # ... 18 total options
    CONCLUDE = "DONE"             # Terminal: output TRUE/FALSE/UNKNOWN
```

Each option has a defined argument count in `OPTION_ARG_COUNTS`:

```python
OPTION_ARG_COUNTS = {
    OptionType.MODUS_PONENS: 2,      # (premise_idx, implication_idx)
    OptionType.AND_ELIM: 2,          # (conjunction_idx, side: 0=left, 1=right)
    OptionType.CONCLUDE: 1,          # (conclusion_type: 0=TRUE, 1=FALSE, 2=UNKNOWN)
    # ...
}
```

### FOLFormula

Represents a first-order logic formula:

```python
@dataclass
class FOLFormula:
    id: int                           # Unique index in the formula set
    nl_text: str                      # Natural language: "All men are mortal"
    fol_string: str                   # FOL: "∀x.(Man(x) → Mortal(x))"
    source: str = "premise"           # "premise" | "derived" | "assumption"
    derived_by: Optional[str] = None  # Which option produced this
    derived_from: list[int] = field(default_factory=list)  # Parent formula IDs
```

### ProofStep

A single reasoning step with Thought/Action format:

```python
@dataclass
class ProofStep:
    step_idx: int                      # Position in the trace
    thought: str                       # NL justification
    option_type: OptionType            # The inference rule
    option_args: list[int]             # Arguments (formula indices)
    result_formula: Optional[FOLFormula] = None
    solver_valid: Optional[bool] = None      # Ground truth from solver
    predicted_valid: Optional[float] = None  # q̂_φ prediction
```

**Key Methods:**

```python
# Generate action string
step.to_action_string()
# -> '<Option type="MODUS_PONENS" args="[0, 1]" />'

# Generate full format
step.to_full_string()
# -> 'Thought: Since we have P and P→Q...\nAction: <Option type="MODUS_PONENS" args="[0, 1]" />'

# Parse from string
ProofStep.from_action_string('<Option type="AND_ELIM" args="[2, 0]" />', step_idx=0)
```

### LogicalState

The current state of a proof:

```python
@dataclass
class LogicalState:
    problem_id: str
    nl_premises: list[str]           # Original NL premises
    fol_formulas: list[FOLFormula]   # Current formula set (premises + derived)
    derived_steps: list[ProofStep]   # History of reasoning steps
    target_conclusion: str           # What to prove
    target_fol: Optional[FOLFormula] # FOL form of conclusion
    label: Optional[str]             # Ground truth: "TRUE" | "FALSE" | "UNKNOWN"
```

**Key Methods:**

```python
state.num_formulas      # Count of current formulas
state.add_formula(f)    # Add derived formula
state.get_formula_by_id(idx)  # Retrieve formula
state.to_prompt()       # Convert to LLM prompt string
```

### OptionizedTrace

A complete proof trace:

```python
@dataclass
class OptionizedTrace:
    problem_id: str
    initial_state: LogicalState
    steps: list[ProofStep]
    final_answer: str                # "TRUE" | "FALSE" | "UNKNOWN"
    trace_valid: Optional[bool]      # All steps valid AND answer correct
```

**Key Properties:**

```python
trace.num_steps              # Number of steps
trace.step_validity_rate     # Fraction of valid steps
trace.to_training_string()   # Full training format
```

### PreferencePair

For DPO training:

```python
@dataclass
class PreferencePair:
    problem_id: str
    prompt: str              # Shared context
    winner: OptionizedTrace  # Solver-valid (preferred)
    loser: OptionizedTrace   # Invalid (dispreferred)
    
    def to_dpo_format(self) -> dict:
        # Returns {"prompt": ..., "chosen": ..., "rejected": ...}
```

---

## 3. Option Vocabulary

### 3.1 Location: `src/data/optionizer.py`

The `Optionizer` class converts raw proofs into optionized format.

### Usage

```python
from src.data.optionizer import Optionizer

optionizer = Optionizer()

# Convert P-FOLIO example
trace = optionizer.optionize_pfolio_example(
    problem_id="folio_001",
    premises=["Socrates is a man", "All men are mortal"],
    fol_premises=["Man(socrates)", "∀x.(Man(x)→Mortal(x))"],
    conclusion="Socrates is mortal",
    fol_conclusion="Mortal(socrates)",
    proof_steps=[
        {"step": "Apply universal instantiation", "rule": "universal instantiation", "from": [1]},
        {"step": "Apply modus ponens", "rule": "modus ponens", "from": [0, 1]},
    ],
    label="TRUE",
)

# Convert PrOntoQA example  
trace = optionizer.optionize_prontoqa_example(
    problem_id="prontoqa_001",
    context="Rex is a cat. All cats are mammals.",
    query="Is Rex a mammal?",
    chain=["Rex is a cat", "All cats are mammals", "So Rex is a mammal"],
    label=True,
)
```

### Rule Mapping

The optionizer maps P-FOLIO rule names to OptionTypes:

```python
PFOLIO_RULE_MAP = {
    "modus ponens": OptionType.MODUS_PONENS,
    "universal instantiation": OptionType.UNIV_INSTANTIATION,
    "conjunction elimination": OptionType.AND_ELIM,
    "proof by cases": OptionType.CASE_SPLIT,
    # ...
}
```

---

## 4. Solvers

### 4.1 Base Interface: `src/solvers/base_solver.py`

All solvers implement the abstract `FOLSolver` interface:

```python
class FOLSolver(ABC):
    @abstractmethod
    def check_step(self, state: LogicalState, step: ProofStep) -> VerificationResult:
        """Verify a single proof step."""
        pass
    
    @abstractmethod
    def check_entailment(self, premises: list[FOLFormula], conclusion: FOLFormula) -> VerificationResult:
        """Check if premises entail conclusion."""
        pass
    
    @abstractmethod
    def check_consistency(self, formulas: list[FOLFormula]) -> VerificationResult:
        """Check if formula set is consistent."""
        pass
    
    def verify_trace(self, trace: OptionizedTrace, ground_truth: str) -> tuple[bool, list]:
        """Verify complete trace (implemented in base class)."""
```

### VerificationResult

```python
class ValidityStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    ERROR = "error"

@dataclass
class VerificationResult:
    status: ValidityStatus
    new_formula: Optional[FOLFormula] = None  # If valid, the derived formula
    message: str = ""
    details: dict = None
    
    @property
    def is_valid(self) -> bool: ...
    
    @property
    def is_invalid(self) -> bool: ...
```

### 4.2 FOLIO Solver: `src/solvers/folio_solver.py`

Uses Z3 theorem prover for verification:

```python
from src.solvers.folio_solver import FOLIOSolver

solver = FOLIOSolver(timeout_ms=5000)

# Verify a step
result = solver.check_step(state, step)
if result.is_valid:
    print(f"Step valid: {result.message}")
    new_formula = result.new_formula

# Verify entailment
result = solver.check_entailment(premises, conclusion)

# Check consistency
result = solver.check_consistency(formulas)
```

**Implemented Verifiers:**

- `_verify_modus_ponens`
- `_verify_modus_tollens`
- `_verify_univ_instantiation`
- `_verify_and_intro`
- `_verify_and_elim`
- `_verify_disjunctive_syllogism`
- `_verify_hypothetical_syllogism`
- `_verify_conclusion`

### 4.3 PrOntoQA Solver: `src/solvers/prontoqa_solver.py`

Uses ontology-based reasoning:

```python
from src.solvers.prontoqa_solver import PrOntoQASolver

solver = PrOntoQASolver()

# Parse context into ontology
solver.parse_context("Rex is a cat. All cats are mammals.")

# Query the ontology
is_mammal = solver.check_query("Rex", "mammal")  # True

# Derive all categories for an entity
categories = solver.derive_categories("rex")  # {"cat", "mammal"}
```

---

## 5. Model Components

### 5.1 Option Success Head: `src/models/option_head.py`

The **q̂_φ(s,ω)** predictor - predicts P(step is valid | state, option).

```python
from src.models.option_head import OptionSuccessHead

# Initialize
head = OptionSuccessHead(
    hidden_dim=4096,          # LLM hidden size
    option_embed_dim=64,      # Option type embedding size
    mlp_hidden_dim=512,
    dropout=0.1,
)

# Forward pass
probs = head(hidden_states, option_type_ids)  # [batch, 1]

# Predict for all options at once
all_probs = head.predict_all_options(hidden_state)  # [num_options]

# Training loss
loss = head.compute_loss(hidden_states, option_type_ids, labels)
```

**Extended Version with Arguments:**

```python
from src.models.option_head import OptionSuccessHeadWithArgs

head = OptionSuccessHeadWithArgs(
    hidden_dim=4096,
    option_embed_dim=64,
    arg_embed_dim=32,
    max_args=4,
    max_formula_idx=100,
)

# Forward with argument indices
probs = head(hidden_states, option_type_ids, option_args)
```

### 5.2 GVF Heads: `src/models/gvf_heads.py`

General Value Functions for auxiliary subtasks:

```python
from src.models.gvf_heads import ConsistencyGVF, GoalProgressGVF, CombinedGVFHead

# Individual heads
consistency_head = ConsistencyGVF(hidden_dim=4096)
goal_head = GoalProgressGVF(hidden_dim=4096)

# Combined head (shared features)
combined = CombinedGVFHead(hidden_dim=4096)
predictions = combined(hidden_state)
# -> {"consistency": ..., "goal_progress": ..., "proof_length": ...}

# Compute targets from a trace
targets = ConsistencyGVF.compute_targets_from_trace(trace)
```

---

## 6. Training Pipeline

### 6.1 Supervised Fine-Tuning: `src/training/sft.py`

Trains the model to generate Thought/Action format.

```python
from src.training.sft import SFTConfig, run_sft_pipeline

config = SFTConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="outputs/sft",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5,
    use_peft=True,
    lora_r=64,
)

# Run complete pipeline
model, tokenizer, trainer = run_sft_pipeline(traces, config)
```

**Components:**

- `OptionizedTraceDataset`: PyTorch dataset for traces
- `prepare_sft_data`: Split into train/val
- `setup_model_and_tokenizer`: Load model with LoRA
- `train_sft`: Run training

### 6.2 DPO Training: `src/training/dpo.py`

Aligns model with solver-induced preferences.

```python
from src.training.dpo import DPOConfig, build_preference_pairs, train_dpo

# Build preference pairs from traces
pairs = build_preference_pairs(
    problems=problems,
    traces_per_problem=traces_dict,
    require_valid_winner=True,
)

# Train
config = DPOConfig(
    beta=0.1,
    learning_rate=5e-6,
    num_epochs=1,
)
trainer = train_dpo(model, tokenizer, pairs, config)
```

### 6.3 OaK Loop: `src/training/oak_loop.py`

The complete iterative training cycle.

```python
from src.training.oak_loop import OaKLoopConfig, OaKLoop, run_oak_pipeline

config = OaKLoopConfig(
    num_iterations=3,
    samples_per_problem=8,
    train_option_head=True,
)

# Run complete pipeline
summary = run_oak_pipeline(
    model=model,
    tokenizer=tokenizer,
    solver=solver,
    train_problems=train_problems,
    val_problems=val_problems,
    config=config,
)
```

**OaK Loop Steps:**

1. **Generate Experience**: Sample traces from current policy
2. **Verify**: Label each step with solver
3. **Update Knowledge**: Train q̂_φ on solver labels
4. **Build Preferences**: Create (valid, invalid) pairs
5. **DPO Update**: Improve policy
6. **Repeat**

**Distributed OaK-DPO notes (Dec 2025):**
- Trace generation now initializes torch.distributed (if WORLD_SIZE>1) so all ranks gather traces before preference building, avoiding mismatched per-rank datasets when DPO starts.
- Preference pairs strip the prompt prefix only once (instead of blanket string replace) to prevent accidental removal when the prompt text appears inside the generated reasoning.
- DPO iteration keeps the base `DPOConfig` immutable across iterations; per-iteration checkpoints write to `.../iter_{k}` without nesting.

---

## 7. Inference

### 7.1 Constrained Decoding: `src/inference/constrained_decode.py`

Ensures valid option syntax.

```python
from src.inference.constrained_decode import OptionConstrainer

constrainer = OptionConstrainer()

# Validate an action string
is_valid, error = constrainer.validate_action('<Option type="MODUS_PONENS" args="[0, 1]" />')

# Fix invalid action
fixed = constrainer.fix_action(invalid_action, num_formulas=10)
```

**Grammar Pattern:**

```ebnf
action      ::= "<Option type=\"" OPTION_TYPE "\" args=\"[" ARGS "]\" />"
OPTION_TYPE ::= "MODUS_PONENS" | "MODUS_TOLLENS" | ...
ARGS        ::= INT ("," INT)*
```

### 7.2 Trace Generation: `src/inference/generate_trace.py`

Generate proof traces from the model.

```python
from src.inference.generate_trace import TraceGenerator, GenerationConfig

config = GenerationConfig(
    max_steps=15,
    temperature=0.7,
    use_constrained_decoding=True,
)

generator = TraceGenerator(model, tokenizer, config)

# Generate traces
traces = generator.generate_trace(state, num_samples=8)

# Generate for multiple problems
results = generator.generate_batch(states, num_samples_per_problem=4)
```

---

## 8. Evaluation

### 8.1 Metrics: `src/evaluation/metrics.py`

```python
from src.evaluation.metrics import (
    compute_accuracy,
    compute_step_validity,
    compute_trace_validity,
    compute_brier_score,
    compute_ece,
    compute_all_metrics,
    format_metrics_report,
)

# Compute all metrics
metrics = compute_all_metrics(traces, labels, q_phi_predictions)

# Print report
print(format_metrics_report(metrics))
```

**Available Metrics:**

| Metric | Function | Description |
|--------|----------|-------------|
| Accuracy | `compute_accuracy` | Final answer correctness |
| Step Validity | `compute_step_validity` | % steps solver-valid |
| Trace Validity | `compute_trace_validity` | % fully valid traces |
| Brier Score | `compute_brier_score` | Calibration (lower=better) |
| ECE | `compute_ece` | Expected calibration error |

### 8.2 Calibration Analysis: `src/evaluation/calibration.py`

```python
from src.evaluation.calibration import CalibrationAnalyzer

analyzer = CalibrationAnalyzer(n_bins=10)

# Add predictions
for pred, label in predictions:
    analyzer.add_prediction(pred, label, metadata={"option_type": "MODUS_PONENS"})

# Compute metrics
metrics = analyzer.compute_metrics()
print(f"Brier: {metrics['brier_score']:.4f}")
print(f"ECE: {metrics['ece']:.4f}")

# Per-option breakdown
per_option = analyzer.compute_per_option_metrics()

# Save/load
analyzer.save("calibration.json")
analyzer = CalibrationAnalyzer.load("calibration.json")
```

---

## 9. Configuration

### 9.1 Model Config: `configs/model.yaml`

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  torch_dtype: "bfloat16"

peft:
  enabled: true
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", ...]

option_head:
  option_embed_dim: 64
  mlp_hidden_dim: 512
```

### 9.2 Training Config: `configs/training.yaml`

```yaml
sft:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5

dpo:
  beta: 0.1
  learning_rate: 5.0e-6

oak_loop:
  num_iterations: 3
  samples_per_problem: 8
```

### 9.3 Evaluation Config: `configs/evaluation.yaml`

```yaml
evaluation:
  greedy: true
  compute_calibration: true

ablations:
  full_model:
    constrained_decoding: true
    use_option_head: true
```

---

## 10. Usage Examples

### Complete Training Pipeline

```python
import torch
from src.data.structures import LogicalState, FOLFormula
from src.solvers.prontoqa_solver import PrOntoQASolver
from src.training.sft import SFTConfig, run_sft_pipeline
from src.training.oak_loop import OaKLoopConfig, run_oak_pipeline

# 1. Prepare data
problems = [
    LogicalState(
        problem_id="001",
        nl_premises=["Rex is a cat", "All cats are mammals"],
        fol_formulas=[...],
        target_conclusion="Rex is a mammal",
        label="TRUE",
    ),
    # ...
]

# 2. Run SFT
sft_config = SFTConfig(model_name="meta-llama/Llama-3.1-8B-Instruct")
model, tokenizer, _ = run_sft_pipeline(traces, sft_config)

# 3. Run OaK-DPO
solver = PrOntoQASolver()
oak_config = OaKLoopConfig(num_iterations=3)
summary = run_oak_pipeline(model, tokenizer, solver, problems, config=oak_config)

# 4. Evaluate
from src.inference.generate_trace import TraceGenerator
from src.evaluation.metrics import compute_all_metrics

generator = TraceGenerator(model, tokenizer)
test_traces = [generator.generate_trace(p, 1)[0] for p in test_problems]

for trace in test_traces:
    solver.verify_trace(trace, trace.initial_state.label)

metrics = compute_all_metrics(test_traces, [p.label for p in test_problems])
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Trace Validity: {metrics['trace_validity']:.2%}")
```

### Using the Option Head for Inference

```python
from src.models.option_head import OptionSuccessHead
from src.data.structures import OptionType

# Load trained head
head = OptionSuccessHead.from_pretrained("checkpoints/option_head.pt", hidden_dim=4096)

# Get hidden state from LLM at decision point
hidden_state = model.get_hidden_state(...)  # [1, hidden_dim]

# Predict success for each option type
all_probs = head.predict_all_options(hidden_state)

# Select best option
best_option_idx = all_probs.argmax()
best_option = list(OptionType)[best_option_idx]
print(f"Best option: {best_option.name} (p={all_probs[best_option_idx]:.3f})")
```

### Custom Solver Integration

```python
from src.solvers.base_solver import FOLSolver, VerificationResult, ValidityStatus

class MyCustomSolver(FOLSolver):
    def check_step(self, state, step):
        # Your verification logic
        is_valid = self._my_verification(state, step)
        return VerificationResult(
            status=ValidityStatus.VALID if is_valid else ValidityStatus.INVALID,
            message="Custom verification",
        )
    
    def check_entailment(self, premises, conclusion):
        # ...
        
    def check_consistency(self, formulas):
        # ...

# Use in training
solver = MyCustomSolver()
summary = run_oak_pipeline(model, tokenizer, solver, problems)
```

---

## Appendix: File Reference

| File | Lines | Description |
|------|-------|-------------|
| `src/data/structures.py` | ~350 | Core data classes and option vocabulary |
| `src/data/optionizer.py` | ~250 | Proof optionization |
| `src/solvers/base_solver.py` | ~120 | Abstract solver interface |
| `src/solvers/folio_solver.py` | ~350 | Z3-based FOLIO solver |
| `src/solvers/prontoqa_solver.py` | ~220 | Ontology-based PrOntoQA solver |
| `src/models/option_head.py` | ~220 | q̂_φ predictor |
| `src/models/gvf_heads.py` | ~280 | GVF auxiliary heads |
| `src/training/sft.py` | ~180 | SFT pipeline |
| `src/training/dpo.py` | ~200 | DPO training |
| `src/training/oak_loop.py` | ~300 | OaK loop orchestration |
| `src/inference/constrained_decode.py` | ~280 | Grammar constraints |
| `src/inference/generate_trace.py` | ~220 | Trace generation |
| `src/evaluation/metrics.py` | ~250 | Evaluation metrics |
| `src/evaluation/calibration.py` | ~220 | Calibration analysis |

**Total: ~3,400 lines of Python**

