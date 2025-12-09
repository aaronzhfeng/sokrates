# SOKRATES Glossary

## Core Concepts

### Options
**Temporally extended actions** that represent reusable reasoning skills. In SOKRATES, options are discrete inference-rule macros (e.g., `MODUS_PONENS`, `UNIV_INSTANTIATION`) rather than raw token sequences. Each option takes specific arguments (indices into the current formula set) and produces a new derived formula.

### Knowledge
**Predictive models about options**. In the OaK framework, knowledge means understanding what options will do before executing them. SOKRATES implements this via:
- **External knowledge**: The FOL solver provides ground-truth about step validity
- **Internal knowledge**: The learned predictor q̂_φ(s,ω) estimates the probability an option will be valid

### OaK (Options and Knowledge)
Rich Sutton's **architectural framework** for building intelligent agents that:
1. Learn reusable behaviors (options)
2. Build predictive models of those behaviors (knowledge)
3. Use knowledge to plan and select options
4. Continuously improve through experience

SOKRATES is a "micro-OaK" instantiation in the logical reasoning domain.

### DPO (Direct Preference Optimization)
An **alignment technique** that fine-tunes language models from preference data without explicit reward modeling. Given pairs of (preferred, dispreferred) outputs, DPO directly optimizes the policy to increase the likelihood gap. SOKRATES uses solver validity to construct these preference pairs.

---

## Reasoning Frameworks

### Chain-of-Thought (CoT)
A prompting/training technique where models generate intermediate reasoning steps before the final answer. Standard CoT is **free-form text**; SOKRATES replaces this with **structured option sequences**.

### Optionized CoT
SOKRATES's structured reasoning format where each proof step consists of:
- **Thought**: Natural language justification
- **Action**: Formal option invocation `<Option type="..." args="[...]" />`

### Constrained Decoding
Restricting the model's token generation to follow a **formal grammar**. SOKRATES uses this to ensure option actions are always syntactically valid (correct option types, valid argument indices).

---

## Datasets

### FOLIO
A **first-order logic reasoning benchmark** with:
- Natural language stories
- FOL formula annotations
- Entailment labels (TRUE/FALSE/UNKNOWN)

### P-FOLIO
An extension of FOLIO with **human-annotated proof chains**:
- Step-by-step reasoning traces
- Inference rule labels for each step
- Source of SOKRATES's option vocabulary

### PrOntoQA
A **synthetic reasoning dataset** based on ontology structures:
- Programmatically generated logical puzzles
- Clean ground-truth verification
- Used for large-scale pretraining in SOKRATES

---

## Inference Rules (Options)

### MODUS_PONENS (MP)
From P and P→Q, derive Q.
> "Socrates is a man" + "All men are mortal" → "Socrates is mortal"

### MODUS_TOLLENS (MT)
From ¬Q and P→Q, derive ¬P.
> "Socrates is not mortal" + "All men are mortal" → "Socrates is not a man"

### UNIV_INSTANTIATION (UI)
From ∀x.P(x), derive P(c) for a specific constant c.
> "All men are mortal" → "Socrates is mortal" (instantiating with Socrates)

### UNIV_GENERALIZATION (UG)
From P(c) for arbitrary c, derive ∀x.P(x).
> If we proved P(c) for an arbitrary c, we can conclude ∀x.P(x)

### EXIST_INSTANTIATION (EI)
From ∃x.P(x), derive P(sk) for a fresh Skolem constant.
> "There exists a philosopher" → "sk₁ is a philosopher"

### EXIST_GENERALIZATION (EG)
From P(c), derive ∃x.P(x).
> "Socrates is a philosopher" → "There exists a philosopher"

### AND_INTRO (AI)
From P and Q, derive P∧Q.
> "Socrates is wise" + "Socrates is Greek" → "Socrates is wise and Greek"

### AND_ELIM (AE)
From P∧Q, derive P (or Q).
> "Socrates is wise and Greek" → "Socrates is wise"

### OR_INTRO (OI)
From P, derive P∨Q.
> "Socrates is Greek" → "Socrates is Greek or Roman"

### DISJUNCTIVE_SYLLOGISM (DS)
From P∨Q and ¬P, derive Q.
> "It's day or night" + "It's not day" → "It's night"

### CASE_SPLIT (CS)
Given P∨Q, reason separately assuming P and assuming Q.
> To prove R from "It's day or night," prove R assuming day, then prove R assuming night

### CONTRADICTION (CN)
From P and ¬P, derive ⊥ (falsity).
> "It's raining" + "It's not raining" → Contradiction

### HYPOTHETICAL_SYLLOGISM (HS)
From P→Q and Q→R, derive P→R.
> "If human, then mortal" + "If mortal, then will die" → "If human, then will die"

---

## Model Components

### Base Model
The pretrained 7-8B parameter instruction-tuned LLM (e.g., LLaMA-3-8B-Instruct, Qwen-2.5-7B-Instruct) that generates optionized reasoning traces.

### LoRA (Low-Rank Adaptation)
A **parameter-efficient fine-tuning** method that adds small trainable matrices to attention layers. Enables training large models on limited hardware.

### Option-Success Head (q̂_φ)
A small neural network that predicts **P(step is solver-valid | state, option)**. Trained on solver labels. Represents SOKRATES's "internal knowledge" about options.

### GVF (General Value Function)
A **predictive model** that estimates expected cumulative signal under some policy. SOKRATES uses simple GVF-style heads for auxiliary subtasks:
- **Consistency GVF**: Will the proof remain contradiction-free?
- **Goal-Progress GVF**: Does this step bring us closer to the target?

---

## Evaluation Metrics

### Final Accuracy
Percentage of problems where the model's final answer (TRUE/FALSE/UNKNOWN) matches the ground truth.

### Step Validity Rate
Percentage of individual proof steps that the solver verifies as logically valid.

### Full-Trace Validity
Percentage of complete proofs where:
1. All steps are solver-valid, AND
2. The final answer is correct

### Brier Score
Measures **calibration** of probabilistic predictions:
> Brier = (1/n) Σ (predicted_prob - actual_outcome)²

Lower is better. Used to evaluate q̂_φ predictions.

### ECE (Expected Calibration Error)
Another calibration metric that bins predictions by confidence and measures the gap between predicted and actual accuracy in each bin. Lower is better.

---

## Related Work

### LoCo-LMs
**Logically Consistent Language Models**. Uses a neuro-symbolic semantic loss to enforce logical constraints during training. Does not use explicit options or option models.

### VeriCoT
**Verified Chain-of-Thought**. Parses CoT into FOL, verifies each step with a solver, and uses this for supervision. Similar verification approach but treats reasoning as unstructured text.

### STOMP (Subtask, Option, Model, Planning)
Sutton's framework where:
- **Subtasks** are reward-respecting goals
- **Options** are policies that achieve subtasks
- **Models** predict option outcomes
- **Planning** uses models to select options

SOKRATES is a simplified STOMP cycle for logical reasoning.

---

## Abbreviations

| Abbrev. | Full Form |
|---------|-----------|
| CoT | Chain-of-Thought |
| DPO | Direct Preference Optimization |
| ECE | Expected Calibration Error |
| FOL | First-Order Logic |
| GVF | General Value Function |
| LM | Language Model |
| LoRA | Low-Rank Adaptation |
| NL | Natural Language |
| OaK | Options and Knowledge |
| PEFT | Parameter-Efficient Fine-Tuning |
| RL | Reinforcement Learning |
| SFT | Supervised Fine-Tuning |
| STOMP | Subtask, Option, Model, Planning |

