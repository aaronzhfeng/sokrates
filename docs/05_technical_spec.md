# SOKRATES Technical Specification

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SOKRATES Architecture                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │   Datasets   │───▶│  Optionizer  │───▶│   Optionized Training Data   │  │
│  │ FOLIO/P-FOLIO│    │              │    │   Thought/Action sequences   │  │
│  │  PrOntoQA    │    └──────────────┘    └──────────────────────────────┘  │
│  └──────────────┘                                      │                    │
│                                                        ▼                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         OaK-DPO Loop                                  │  │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌────────┐ │  │
│  │  │ Generate│──▶│ Verify  │──▶│ Update  │──▶│  Build  │──▶│  DPO   │ │  │
│  │  │ Traces  │   │ (Solver)│   │  q̂_φ    │   │  Prefs  │   │ Train  │ │  │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └────────┘ │  │
│  │       ▲                                                       │      │  │
│  │       └───────────────────────────────────────────────────────┘      │  │
│  │                         (2-3 iterations)                             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Inference Pipeline                             │  │
│  │  Input ──▶ Constrained Decode ──▶ Option Sequence ──▶ Final Answer   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Option Vocabulary

### 2.1 Core Inference Rules (from P-FOLIO taxonomy)

| Option ID | Name | Arguments | Description |
|-----------|------|-----------|-------------|
| `MP` | `MODUS_PONENS` | `(premise_idx, implication_idx)` | From P and P→Q, derive Q |
| `MT` | `MODUS_TOLLENS` | `(negated_consequent_idx, implication_idx)` | From ¬Q and P→Q, derive ¬P |
| `UI` | `UNIV_INSTANTIATION` | `(formula_idx, constant)` | From ∀x.P(x), derive P(c) |
| `UG` | `UNIV_GENERALIZATION` | `(formula_idx, variable)` | From P(c) for arbitrary c, derive ∀x.P(x) |
| `EI` | `EXIST_INSTANTIATION` | `(formula_idx, skolem_constant)` | From ∃x.P(x), derive P(sk) |
| `EG` | `EXIST_GENERALIZATION` | `(formula_idx, variable)` | From P(c), derive ∃x.P(x) |
| `AI` | `AND_INTRO` | `(formula_idx_1, formula_idx_2)` | From P and Q, derive P∧Q |
| `AE` | `AND_ELIM` | `(conjunction_idx, side)` | From P∧Q, derive P or Q |
| `OI` | `OR_INTRO` | `(formula_idx, new_disjunct)` | From P, derive P∨Q |
| `DS` | `DISJUNCTIVE_SYLLOGISM` | `(disjunction_idx, negated_disjunct_idx)` | From P∨Q and ¬P, derive Q |
| `CS` | `CASE_SPLIT` | `(disjunction_idx)` | Split into cases for P∨Q |
| `CN` | `CONTRADICTION` | `(formula_idx, negation_idx)` | From P and ¬P, derive ⊥ |
| `DN` | `DOUBLE_NEGATION` | `(formula_idx)` | From ¬¬P, derive P |
| `CP` | `CONDITIONAL_PROOF` | `(assumption_idx, conclusion_idx)` | Assuming P, derived Q → conclude P→Q |
| `HS` | `HYPOTHETICAL_SYLLOGISM` | `(impl_1_idx, impl_2_idx)` | From P→Q and Q→R, derive P→R |
| `BI` | `BICONDITIONAL_INTRO` | `(impl_1_idx, impl_2_idx)` | From P→Q and Q→P, derive P↔Q |
| `BE` | `BICONDITIONAL_ELIM` | `(biconditional_idx, direction)` | From P↔Q, derive P→Q or Q→P |
| `DONE` | `CONCLUDE` | `(conclusion_type)` | Terminal: output TRUE/FALSE/UNKNOWN |

### 2.2 Option Token Format

```
<Option type="MODUS_PONENS" args="[2, 5]" />
```

Full step format (Thought/Action):
```
Thought: Since we know "Socrates is a man" (premise 2) and "All men are mortal" (premise 5), 
we can apply modus ponens to conclude that Socrates is mortal.
Action: <Option type="MODUS_PONENS" args="[2, 5]" />
```

---

## 3. Data Structures

### 3.1 Logical State

```python
@dataclass
class LogicalState:
    """Represents the current state of a proof."""
    problem_id: str
    nl_premises: List[str]           # Natural language premises
    fol_formulas: List[FOLFormula]   # Current derived FOL formulas
    derived_steps: List[ProofStep]   # History of applied options
    target_conclusion: str           # What we're trying to prove
    target_fol: FOLFormula           # FOL form of conclusion
    
@dataclass
class FOLFormula:
    """First-order logic formula."""
    id: int
    nl_text: str                     # Natural language gloss
    fol_string: str                  # FOL syntax string
    source: str                      # "premise" | "derived"
    derived_by: Optional[str]        # Option that derived this
```

### 3.2 Proof Step

```python
@dataclass
class ProofStep:
    """A single reasoning step."""
    step_idx: int
    thought: str                     # NL justification
    option_type: str                 # e.g., "MODUS_PONENS"
    option_args: List[int]           # Indices into formula list
    result_formula: Optional[FOLFormula]
    solver_valid: Optional[bool]     # Ground truth from solver
    predicted_valid: Optional[float] # q̂_φ prediction
```

### 3.3 Optionized Trace

```python
@dataclass
class OptionizedTrace:
    """Complete proof trace."""
    problem_id: str
    initial_state: LogicalState
    steps: List[ProofStep]
    final_answer: str                # "TRUE" | "FALSE" | "UNKNOWN"
    trace_valid: bool                # All steps valid AND answer correct
    
@dataclass
class PreferencePair:
    """DPO training pair."""
    problem_id: str
    prompt: str                      # Shared context
    winner: OptionizedTrace          # Solver-valid trace
    loser: OptionizedTrace           # Invalid trace
```

---

## 4. Solver Interface

### 4.1 Abstract Solver API

```python
class FOLSolver(ABC):
    """Abstract interface for FOL verification."""
    
    @abstractmethod
    def check_step(
        self, 
        state: LogicalState, 
        option: ProofStep
    ) -> Tuple[bool, Optional[FOLFormula], str]:
        """
        Verify a single proof step.
        
        Returns:
            (is_valid, new_formula_if_valid, error_message)
        """
        pass
    
    @abstractmethod
    def check_entailment(
        self,
        premises: List[FOLFormula],
        conclusion: FOLFormula
    ) -> bool:
        """Check if premises entail conclusion."""
        pass
    
    @abstractmethod
    def check_consistency(
        self,
        formulas: List[FOLFormula]
    ) -> bool:
        """Check if formula set is consistent (no contradictions)."""
        pass
```

### 4.2 Solver Implementations

```python
# For FOLIO/P-FOLIO (uses Prover9/Mace4 or Z3)
class FOLIOSolver(FOLSolver):
    """Solver using FOLIO's FOL annotations."""
    pass

# For PrOntoQA (uses built-in ontology engine)
class PrOntoQASolver(FOLSolver):
    """Solver using PrOntoQA's synthetic world model."""
    pass
```

---

## 5. Model Architecture

### 5.1 Base Model Configuration

```yaml
base_model:
  name: "meta-llama/Llama-3-8B-Instruct"  # or Qwen-2.5-7B-Instruct
  quantization: null  # Full precision for training
  
peft:
  method: "lora"
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  lora_dropout: 0.05
```

### 5.2 Option-Success Predictor Head (q̂_φ)

```python
class OptionSuccessHead(nn.Module):
    """
    Predicts P(step is solver-valid | state, option).
    Attached to base LM's hidden states.
    """
    def __init__(self, hidden_dim: int, num_option_types: int):
        super().__init__()
        self.option_embed = nn.Embedding(num_option_types, 64)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_state: Tensor, option_type_id: int) -> Tensor:
        """
        Args:
            hidden_state: [batch, hidden_dim] from LM at decision point
            option_type_id: Index of option type
        Returns:
            Predicted success probability [batch, 1]
        """
        opt_emb = self.option_embed(option_type_id)
        combined = torch.cat([hidden_state, opt_emb], dim=-1)
        return self.mlp(combined)
```

### 5.3 GVF Heads (Auxiliary Subtasks)

```python
class ConsistencyGVF(nn.Module):
    """Predicts: will the proof remain contradiction-free?"""
    # Similar architecture to OptionSuccessHead
    pass

class GoalProgressGVF(nn.Module):
    """Predicts: does this step bring us closer to the target?"""
    pass
```

---

## 6. Constrained Decoding

### 6.1 Grammar for Option Actions

```ebnf
action      ::= "<Option " "type=\"" option_type "\" " "args=\"" args "\" />"
option_type ::= "MODUS_PONENS" | "MODUS_TOLLENS" | "UNIV_INSTANTIATION" | ...
args        ::= "[" arg_list "]"
arg_list    ::= integer | integer "," arg_list
integer     ::= [0-9]+
```

### 6.2 Outlines Integration

```python
from outlines import models, generate

# Define grammar constraint
option_grammar = r"""
start: "<Option type=\"" OPTION_TYPE "\" args=\"[" ARGS "]\" />"
OPTION_TYPE: "MODUS_PONENS" | "MODUS_TOLLENS" | "UNIV_INSTANTIATION" | ...
ARGS: INT ("," INT)*
INT: /[0-9]+/
"""

# Create constrained generator
model = models.transformers("meta-llama/Llama-3-8B-Instruct")
option_generator = generate.cfg(model, option_grammar)

# During inference
def generate_action(prompt: str, state: LogicalState) -> str:
    """Generate syntactically valid option action."""
    return option_generator(prompt)
```

---

## 7. Training Pipeline

### 7.1 Stage 1: Supervised Fine-Tuning

```python
# Convert P-FOLIO proofs to Thought/Action format
training_config_sft = {
    "dataset": "p-folio-optionized",
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation": 8,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "max_seq_length": 2048,
}
```

### 7.2 Stage 2: OaK-DPO Loop

```python
def oak_dpo_iteration(model, dataset, solver, iteration: int):
    """One iteration of the OaK-DPO loop."""
    
    # 1. Generate traces
    traces = []
    for problem in dataset:
        for _ in range(K_SAMPLES):  # K=8-16
            trace = generate_optionized_trace(model, problem)
            traces.append(trace)
    
    # 2. Verify with solver
    for trace in traces:
        for step in trace.steps:
            step.solver_valid = solver.check_step(trace.state_at(step), step)
        trace.trace_valid = all(s.solver_valid for s in trace.steps) and trace.answer_correct
    
    # 3. Update option-success head
    train_option_head(model.option_head, traces)
    
    # 4. Build preference pairs
    preferences = build_preference_pairs(traces)
    
    # 5. DPO training
    dpo_train(model, preferences, config=dpo_config)
    
    return model

# Run 2-3 iterations
for i in range(NUM_OAK_ITERATIONS):
    model = oak_dpo_iteration(model, train_data, solver, i)
```

### 7.3 DPO Configuration

```python
dpo_config = {
    "beta": 0.1,                    # KL penalty coefficient
    "loss_type": "sigmoid",         # Standard DPO loss
    "learning_rate": 5e-6,
    "epochs": 1,
    "batch_size": 2,
    "gradient_accumulation": 16,
    "max_length": 2048,
    "max_prompt_length": 1024,
}
```

---

## 8. Evaluation Protocol

### 8.1 Metrics Implementation

```python
def evaluate(model, test_set, solver) -> Dict[str, float]:
    results = {
        "accuracy": [],           # Final answer correct
        "step_validity": [],      # % steps valid per trace
        "trace_validity": [],     # Full trace valid
        "q_phi_brier": [],        # Brier score for q̂_φ
        "q_phi_ece": [],          # Expected calibration error
    }
    
    for problem in test_set:
        trace = generate_optionized_trace(model, problem, greedy=True)
        
        # Verify
        step_results = [solver.check_step(s) for s in trace.steps]
        predictions = [model.option_head(s) for s in trace.steps]
        
        # Compute metrics
        results["accuracy"].append(trace.final_answer == problem.label)
        results["step_validity"].append(mean(step_results))
        results["trace_validity"].append(all(step_results) and trace.answer_correct)
        results["q_phi_brier"].append(brier_score(predictions, step_results))
        results["q_phi_ece"].append(expected_calibration_error(predictions, step_results))
    
    return {k: mean(v) for k, v in results.items()}
```

### 8.2 Ablation Configurations

| Ablation | Description |
|----------|-------------|
| `no_constrained` | Remove grammar constraints on option decoding |
| `no_option_head` | Remove q̂_φ predictor, use only DPO |
| `raw_cot_dpo` | DPO on free-form CoT instead of optionized |
| `no_oak_loop` | Single DPO pass (no iterations) |
| `answer_only_pref` | Preferences based only on final answer, not step validity |

---

## 9. File Structure

```
sokrates/
├── configs/
│   ├── model.yaml
│   ├── training.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/
│   │   ├── folio/
│   │   ├── p_folio/
│   │   └── prontoqa/
│   └── processed/
│       ├── optionized_train.jsonl
│       ├── optionized_dev.jsonl
│       └── preferences/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── optionizer.py          # Convert proofs to Thought/Action
│   │   ├── folio_loader.py
│   │   ├── prontoqa_loader.py
│   │   └── preference_builder.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── option_head.py         # q̂_φ implementation
│   │   └── gvf_heads.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base_solver.py
│   │   ├── folio_solver.py
│   │   └── prontoqa_solver.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft.py
│   │   ├── dpo.py
│   │   └── oak_loop.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── constrained_decode.py
│   │   └── generate_trace.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── calibration.py
├── scripts/
│   ├── prepare_data.py
│   ├── train_sft.py
│   ├── run_oak_dpo.py
│   └── evaluate.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── results_analysis.ipynb
├── tests/
│   └── ...
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 10. Dependencies

```txt
# Core
torch>=2.1.0
transformers>=4.36.0
peft>=0.7.0
trl>=0.7.0
datasets>=2.15.0

# Constrained Decoding
outlines>=0.0.34

# FOL Solvers
z3-solver>=4.12.0

# Experiment Tracking
wandb>=0.16.0

# Utilities
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.66.0
pyyaml>=6.0.0
jsonlines>=4.0.0

# Development
pytest>=7.4.0
black>=23.0.0
ruff>=0.1.0
```

---

## 11. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× RTX 4090 (24GB) | 4× A40 (48GB each) |
| VRAM | 24GB | 48GB+ |
| RAM | 64GB | 128GB |
| Storage | 100GB SSD | 500GB NVMe |

### Training Time Estimates (4× A40)

| Phase | Duration |
|-------|----------|
| SFT (7-8B, LoRA) | ~4-6 hours |
| Trace generation (50k traces) | ~8-12 hours |
| DPO per iteration | ~2-4 hours |
| Full OaK loop (3 iterations) | ~24-36 hours |
| Evaluation | ~1-2 hours |

