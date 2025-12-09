# SOKRATES API Reference

Complete API documentation for all public classes and functions.

---

## Data Module (`src.data`)

### structures.py

#### `class OptionType(Enum)`

Enumeration of inference rule options.

**Members:**

| Name | Value | Args | Description |
|------|-------|------|-------------|
| `MODUS_PONENS` | `"MP"` | 2 | From P and P→Q, derive Q |
| `MODUS_TOLLENS` | `"MT"` | 2 | From ¬Q and P→Q, derive ¬P |
| `HYPOTHETICAL_SYLLOGISM` | `"HS"` | 2 | From P→Q and Q→R, derive P→R |
| `DISJUNCTIVE_SYLLOGISM` | `"DS"` | 2 | From P∨Q and ¬P, derive Q |
| `UNIV_INSTANTIATION` | `"UI"` | 2 | From ∀x.P(x), derive P(c) |
| `UNIV_GENERALIZATION` | `"UG"` | 2 | From P(c), derive ∀x.P(x) |
| `EXIST_INSTANTIATION` | `"EI"` | 2 | From ∃x.P(x), derive P(sk) |
| `EXIST_GENERALIZATION` | `"EG"` | 2 | From P(c), derive ∃x.P(x) |
| `AND_INTRO` | `"AI"` | 2 | From P and Q, derive P∧Q |
| `AND_ELIM` | `"AE"` | 2 | From P∧Q, derive P or Q |
| `OR_INTRO` | `"OI"` | 2 | From P, derive P∨Q |
| `CASE_SPLIT` | `"CS"` | 1 | Split into cases for P∨Q |
| `BICONDITIONAL_INTRO` | `"BI"` | 2 | From P→Q and Q→P, derive P↔Q |
| `BICONDITIONAL_ELIM` | `"BE"` | 2 | From P↔Q, derive P→Q |
| `CONTRADICTION` | `"CN"` | 2 | From P and ¬P, derive ⊥ |
| `DOUBLE_NEGATION` | `"DN"` | 1 | From ¬¬P, derive P |
| `CONDITIONAL_PROOF` | `"CP"` | 2 | Assuming P→Q, derive P→Q |
| `CONCLUDE` | `"DONE"` | 1 | Terminal step |

---

#### `class FOLFormula`

First-order logic formula representation.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `id` | `int` | Unique identifier |
| `nl_text` | `str` | Natural language text |
| `fol_string` | `str` | FOL syntax string |
| `source` | `str` | `"premise"`, `"derived"`, or `"assumption"` |
| `derived_by` | `Optional[str]` | Option that produced this |
| `derived_from` | `list[int]` | Parent formula IDs |

**Methods:**

```python
def __str__(self) -> str
    """Returns: '[id] nl_text | fol_string'"""

def __repr__(self) -> str
    """Returns: 'FOLFormula(id=..., fol=...)'"""
```

---

#### `class ProofStep`

Single reasoning step with Thought/Action format.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `step_idx` | `int` | Position in trace |
| `thought` | `str` | Natural language justification |
| `option_type` | `OptionType` | Inference rule |
| `option_args` | `list[int]` | Arguments |
| `result_formula` | `Optional[FOLFormula]` | Derived formula |
| `solver_valid` | `Optional[bool]` | Solver verification result |
| `predicted_valid` | `Optional[float]` | q̂_φ prediction |

**Methods:**

```python
def to_action_string(self) -> str
    """Convert to '<Option type="..." args="[...]" />' format."""

def to_full_string(self) -> str
    """Convert to 'Thought: ...\nAction: ...' format."""

@classmethod
def from_action_string(cls, action_str: str, step_idx: int, thought: str = "") -> ProofStep
    """Parse action string into ProofStep."""
```

---

#### `class LogicalState`

Current state of a proof.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `problem_id` | `str` | Unique identifier |
| `nl_premises` | `list[str]` | Natural language premises |
| `fol_formulas` | `list[FOLFormula]` | Current formula set |
| `derived_steps` | `list[ProofStep]` | Step history |
| `target_conclusion` | `str` | Conclusion to prove |
| `target_fol` | `Optional[FOLFormula]` | FOL conclusion |
| `label` | `Optional[str]` | Ground truth |

**Properties:**

```python
@property
def num_formulas(self) -> int
    """Number of formulas in current state."""

@property
def num_steps(self) -> int
    """Number of derived steps."""
```

**Methods:**

```python
def add_formula(self, formula: FOLFormula) -> None
    """Add formula to state."""

def add_step(self, step: ProofStep) -> None
    """Add step and its result formula."""

def get_formula_by_id(self, idx: int) -> Optional[FOLFormula]
    """Get formula by index, None if out of range."""

def to_prompt(self) -> str
    """Convert to LLM prompt string."""
```

---

#### `class OptionizedTrace`

Complete proof trace.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `problem_id` | `str` | Problem identifier |
| `initial_state` | `LogicalState` | Starting state |
| `steps` | `list[ProofStep]` | Proof steps |
| `final_answer` | `str` | `"TRUE"`, `"FALSE"`, or `"UNKNOWN"` |
| `trace_valid` | `Optional[bool]` | Overall validity |

**Properties:**

```python
@property
def num_steps(self) -> int
    """Number of steps in trace."""

@property
def step_validity_rate(self) -> float
    """Fraction of valid steps."""
```

**Methods:**

```python
def to_training_string(self) -> str
    """Convert to full training format."""
```

---

#### `class PreferencePair`

DPO training pair.

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `problem_id` | `str` | Problem identifier |
| `prompt` | `str` | Shared context |
| `winner` | `OptionizedTrace` | Preferred (valid) trace |
| `loser` | `OptionizedTrace` | Dispreferred (invalid) trace |

**Methods:**

```python
def to_dpo_format(self) -> dict
    """Returns: {'prompt': ..., 'chosen': ..., 'rejected': ...}"""
```

---

### optionizer.py

#### `class Optionizer`

Converts raw proofs to optionized format.

**Methods:**

```python
def optionize_pfolio_example(
    self,
    problem_id: str,
    premises: list[str],
    fol_premises: list[str],
    conclusion: str,
    fol_conclusion: str,
    proof_steps: list[dict],
    label: str,
) -> OptionizedTrace
    """
    Convert P-FOLIO example with proof annotations.
    
    Args:
        problem_id: Unique identifier
        premises: NL premise strings
        fol_premises: FOL premise strings
        conclusion: NL conclusion
        fol_conclusion: FOL conclusion
        proof_steps: List of dicts with 'step', 'rule', 'from', 'result'
        label: Ground truth label
    
    Returns:
        OptionizedTrace with structured steps
    """

def optionize_prontoqa_example(
    self,
    problem_id: str,
    context: str,
    query: str,
    chain: list[str],
    label: bool,
) -> OptionizedTrace
    """
    Convert PrOntoQA example.
    
    Args:
        problem_id: Unique identifier
        context: Ontology context
        query: Question to answer
        chain: Reasoning chain
        label: True/False answer
    
    Returns:
        OptionizedTrace
    """

def parse_trace_string(self, trace_str: str, problem_id: str = "") -> OptionizedTrace
    """Parse model output back into OptionizedTrace."""
```

---

## Models Module (`src.models`)

### option_head.py

#### `class OptionSuccessHead(nn.Module)`

Predicts P(step is valid | state, option).

**Constructor:**

```python
def __init__(
    self,
    hidden_dim: int,              # LLM hidden size
    option_embed_dim: int = 64,   # Option embedding dimension
    mlp_hidden_dim: int = 512,    # MLP hidden dimension
    num_option_types: int = None, # Defaults to vocabulary size
    dropout: float = 0.1,
)
```

**Methods:**

```python
def forward(
    self,
    hidden_state: torch.Tensor,    # [batch, hidden_dim]
    option_type_ids: torch.Tensor, # [batch] or [batch, 1]
) -> torch.Tensor
    """
    Predict success probability.
    
    Returns:
        [batch, 1] probabilities in [0, 1]
    """

def predict_all_options(
    self,
    hidden_state: torch.Tensor,    # [batch, hidden_dim] or [hidden_dim]
) -> torch.Tensor
    """
    Predict for all option types.
    
    Returns:
        [batch, num_options] or [num_options] probabilities
    """

def compute_loss(
    self,
    hidden_states: torch.Tensor,
    option_type_ids: torch.Tensor,
    labels: torch.Tensor,          # Binary: 1=valid, 0=invalid
    reduction: str = "mean",
) -> torch.Tensor
    """Compute binary cross-entropy loss."""

@classmethod
def from_pretrained(cls, path: str, hidden_dim: int) -> OptionSuccessHead
    """Load from checkpoint."""

def save(self, path: str)
    """Save to checkpoint."""
```

---

#### `class OptionSuccessHeadWithArgs(OptionSuccessHead)`

Extended head that also considers option arguments.

**Constructor:**

```python
def __init__(
    self,
    hidden_dim: int,
    option_embed_dim: int = 64,
    arg_embed_dim: int = 32,       # Argument embedding dimension
    max_args: int = 4,             # Max arguments to encode
    max_formula_idx: int = 100,    # Max formula index
    mlp_hidden_dim: int = 512,
    num_option_types: int = None,
    dropout: float = 0.1,
)
```

**Methods:**

```python
def forward(
    self,
    hidden_state: torch.Tensor,
    option_type_ids: torch.Tensor,
    option_args: Optional[torch.Tensor] = None,  # [batch, max_args]
) -> torch.Tensor
    """Predict with arguments."""
```

---

### gvf_heads.py

#### `class ConsistencyGVF(BaseGVFHead)`

Predicts: will the proof remain contradiction-free?

**Methods:**

```python
@staticmethod
def compute_targets_from_trace(trace) -> list[float]
    """
    Compute consistency targets for each step.
    
    Target = 1 if all remaining steps are valid, 0 otherwise.
    """
```

---

#### `class GoalProgressGVF(BaseGVFHead)`

Predicts: does this step bring us closer to the goal?

**Methods:**

```python
@staticmethod
def compute_targets_from_trace(
    trace,
    target_predicates: Optional[set] = None,
) -> list[float]
    """Compute goal progress targets based on predicate overlap."""
```

---

#### `class CombinedGVFHead(nn.Module)`

Combines multiple GVF heads with shared features.

**Methods:**

```python
def forward(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]
    """
    Returns:
        {
            "consistency": [batch, 1],
            "goal_progress": [batch, 1],
            "proof_length": [batch, 1],
        }
    """

def compute_loss(
    self,
    hidden_states: torch.Tensor,
    targets: dict[str, torch.Tensor],
    weights: Optional[dict[str, float]] = None,
) -> torch.Tensor
    """Combined loss for all GVFs."""
```

---

## Solvers Module (`src.solvers`)

### base_solver.py

#### `class ValidityStatus(Enum)`

| Member | Description |
|--------|-------------|
| `VALID` | Step/entailment is valid |
| `INVALID` | Step/entailment is invalid |
| `UNKNOWN` | Solver couldn't determine |
| `ERROR` | Solver error |

---

#### `class VerificationResult`

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `status` | `ValidityStatus` | Verification result |
| `new_formula` | `Optional[FOLFormula]` | Derived formula if valid |
| `message` | `str` | Human-readable explanation |
| `details` | `dict` | Additional info |

**Properties:**

```python
@property
def is_valid(self) -> bool

@property
def is_invalid(self) -> bool
```

---

#### `class FOLSolver(ABC)`

Abstract solver interface.

**Abstract Methods:**

```python
@abstractmethod
def check_step(
    self,
    state: LogicalState,
    step: ProofStep,
) -> VerificationResult
    """Verify single proof step."""

@abstractmethod
def check_entailment(
    self,
    premises: list[FOLFormula],
    conclusion: FOLFormula,
) -> VerificationResult
    """Check if premises entail conclusion."""

@abstractmethod
def check_consistency(
    self,
    formulas: list[FOLFormula],
) -> VerificationResult
    """Check formula set consistency."""
```

**Concrete Methods:**

```python
def verify_trace(
    self,
    trace: OptionizedTrace,
    ground_truth_label: Optional[str] = None,
) -> tuple[bool, list[VerificationResult]]
    """
    Verify complete trace.
    
    Returns:
        (overall_valid, list of per-step results)
    """

def get_solver_name(self) -> str
    """Return solver class name."""
```

---

### folio_solver.py

#### `class FOLIOSolver(FOLSolver)`

Z3-based solver for FOLIO/P-FOLIO.

**Constructor:**

```python
def __init__(self, timeout_ms: int = 5000)
    """
    Args:
        timeout_ms: Z3 timeout in milliseconds
    """
```

---

### prontoqa_solver.py

#### `class PrOntoQASolver(FOLSolver)`

Ontology-based solver for PrOntoQA.

**Methods:**

```python
def parse_context(self, context: str) -> None
    """Parse context into ontology facts and rules."""

def derive_categories(self, entity: str) -> set[str]
    """Derive all categories for entity via forward chaining."""

def check_query(self, entity: str, category: str) -> bool
    """Check if entity belongs to category."""
```

---

## Training Module (`src.training`)

### sft.py

#### `class SFTConfig`

**Attributes:**

| Name | Default | Description |
|------|---------|-------------|
| `model_name` | `"meta-llama/Llama-3.1-8B-Instruct"` | Base model |
| `use_peft` | `True` | Use LoRA |
| `lora_r` | `64` | LoRA rank |
| `lora_alpha` | `128` | LoRA alpha |
| `output_dir` | `"outputs/sft"` | Output directory |
| `num_epochs` | `3` | Training epochs |
| `batch_size` | `4` | Batch size |
| `learning_rate` | `2e-5` | Learning rate |
| `max_seq_length` | `2048` | Max sequence length |

---

#### Functions

```python
def prepare_sft_data(
    traces: list[OptionizedTrace],
    tokenizer,
    train_ratio: float = 0.9,
    max_length: int = 2048,
) -> tuple[Dataset, Dataset]
    """Prepare train/val datasets."""

def setup_model_and_tokenizer(config: SFTConfig) -> tuple
    """Set up model with LoRA and tokenizer."""

def train_sft(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[SFTConfig] = None,
) -> Trainer
    """Run SFT training."""

def run_sft_pipeline(
    traces: list[OptionizedTrace],
    config: Optional[SFTConfig] = None,
) -> tuple[model, tokenizer, trainer]
    """Complete SFT pipeline."""
```

---

### dpo.py

#### `class DPOConfig`

**Attributes:**

| Name | Default | Description |
|------|---------|-------------|
| `beta` | `0.1` | KL penalty coefficient |
| `loss_type` | `"sigmoid"` | Loss type |
| `output_dir` | `"outputs/dpo"` | Output directory |
| `num_epochs` | `1` | Training epochs |
| `batch_size` | `2` | Batch size |
| `learning_rate` | `5e-6` | Learning rate |

---

#### Functions

```python
def build_preference_pairs(
    problems: list[LogicalState],
    traces_per_problem: dict[str, list[OptionizedTrace]],
    require_valid_winner: bool = True,
) -> list[PreferencePair]
    """Build preference pairs from traces."""

def train_dpo(
    model,
    tokenizer,
    train_pairs: list[PreferencePair],
    config: Optional[DPOConfig] = None,
    ref_model = None,
) -> Trainer
    """Run DPO training."""
```

---

### oak_loop.py

#### `class OaKLoopConfig`

**Attributes:**

| Name | Default | Description |
|------|---------|-------------|
| `num_iterations` | `3` | OaK loop iterations |
| `samples_per_problem` | `8` | Traces per problem |
| `output_dir` | `"outputs/oak_loop"` | Output directory |
| `train_option_head` | `True` | Train q̂_φ |

---

#### `class OaKLoop`

**Constructor:**

```python
def __init__(
    self,
    model,
    tokenizer,
    solver: FOLSolver,
    config: Optional[OaKLoopConfig] = None,
    option_head: Optional[OptionSuccessHead] = None,
)
```

**Methods:**

```python
def run(
    self,
    train_problems: list[LogicalState],
    val_problems: Optional[list[LogicalState]] = None,
) -> dict
    """Run complete OaK loop. Returns training summary."""
```

---

#### Functions

```python
def run_oak_pipeline(
    model,
    tokenizer,
    solver: FOLSolver,
    train_problems: list[LogicalState],
    val_problems: Optional[list[LogicalState]] = None,
    config: Optional[OaKLoopConfig] = None,
) -> dict
    """Complete OaK pipeline entry point."""
```

---

## Inference Module (`src.inference`)

### constrained_decode.py

#### `class OptionConstrainer`

**Constructor:**

```python
def __init__(self, config: Optional[ConstrainedDecodingConfig] = None)
```

**Methods:**

```python
def validate_action(self, action_str: str) -> tuple[bool, str]
    """Validate action string. Returns (is_valid, error_message)."""

def fix_action(self, action_str: str, num_formulas: int = 10) -> str
    """Attempt to fix invalid action string."""
```

---

### generate_trace.py

#### `class GenerationConfig`

**Attributes:**

| Name | Default | Description |
|------|---------|-------------|
| `max_steps` | `15` | Maximum proof steps |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.9` | Top-p sampling |
| `do_sample` | `True` | Enable sampling |
| `use_constrained_decoding` | `True` | Use grammar constraints |

---

#### `class TraceGenerator`

**Constructor:**

```python
def __init__(
    self,
    model,
    tokenizer,
    config: Optional[GenerationConfig] = None,
    constrainer: Optional[OptionConstrainer] = None,
    device: str = "cuda",
)
```

**Methods:**

```python
def generate_trace(
    self,
    state: LogicalState,
    num_samples: int = 1,
) -> list[OptionizedTrace]
    """Generate traces for a problem."""

def generate_batch(
    self,
    states: list[LogicalState],
    num_samples_per_problem: int = 1,
) -> dict[str, list[OptionizedTrace]]
    """Generate for multiple problems."""
```

---

## Evaluation Module (`src.evaluation`)

### metrics.py

```python
def compute_accuracy(
    predictions: list[str],
    labels: list[str],
) -> float
    """Final answer accuracy."""

def compute_step_validity(traces: list) -> dict[str, float]
    """
    Returns:
        {
            "overall": float,
            "per_trace_mean": float,
            "per_option_type": dict[str, float],
            "total_steps": int,
            "valid_steps": int,
        }
    """

def compute_trace_validity(
    traces: list,
    labels: Optional[list[str]] = None,
) -> float
    """Full trace validity rate."""

def compute_brier_score(
    predictions: list[float],
    labels: list[int],
) -> float
    """Brier score for calibration."""

def compute_ece(
    predictions: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> float
    """Expected Calibration Error."""

def compute_all_metrics(
    traces: list,
    labels: list[str],
    q_phi_predictions: Optional[list[tuple]] = None,
) -> dict
    """Compute all metrics."""

def format_metrics_report(metrics: dict) -> str
    """Format as human-readable report."""
```

---

### calibration.py

#### `class CalibrationAnalyzer`

**Constructor:**

```python
def __init__(self, n_bins: int = 10)
```

**Methods:**

```python
def add_prediction(
    self,
    prediction: float,
    label: int,
    metadata: Optional[dict] = None,
)
    """Add single prediction."""

def add_batch(
    self,
    predictions: list[float],
    labels: list[int],
    metadata: Optional[list[dict]] = None,
)
    """Add multiple predictions."""

def compute_metrics(self) -> dict
    """
    Returns:
        {
            "brier_score": float,
            "ece": float,
            "calibration_curve": dict,
            "num_samples": int,
        }
    """

def compute_per_option_metrics(self) -> dict[str, dict]
    """Metrics per option type."""

def save(self, path: str)
    """Save to JSON."""

@classmethod
def load(cls, path: str) -> CalibrationAnalyzer
    """Load from JSON."""
```

