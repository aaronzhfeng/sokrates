# 04: SOKRATES Pipeline Guide

Complete technical guide to the SOKRATES training pipeline, data formats, and how each component connects.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SOKRATES Training Pipeline                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Raw Data (PrOntoQA)                                                    │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────┐     Learns Thought/Action format                          │
│  │   SFT   │     with option vocabulary                                │
│  └────┬────┘                                                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     OaK Loop (2-3 iterations)                    │   │
│  │  ┌──────────┐    ┌──────────┐    ┌─────────┐    ┌───────────┐   │   │
│  │  │ Generate │───▶│  Verify  │───▶│  Build  │───▶│    DPO    │   │   │
│  │  │  Traces  │    │ (Solver) │    │  Pairs  │    │  Training │   │   │
│  │  └──────────┘    └──────────┘    └─────────┘    └─────┬─────┘   │   │
│  │       ▲                                               │         │   │
│  │       └───────────────────────────────────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  Final Model (DPO-aligned)                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Raw Data Structure

### PrOntoQA Format

Each problem in `data/processed/prontoqa_train.jsonl`:

```json
{
    "problem_id": "ProntoQA_train_hop-1_case-257_ic-5",
    "training_text": "Premises:\n  [0] Wren is a jompus\n  [1] Rompuses are not spicy\n  [2] Each rompus is a dumpus\n  ...\n\nConclusion to evaluate: Wren is not Jompus.\n\nReasoning:\nThought: Nervous('Wren', True)\nAction: <Option type=\"MODUS_PONENS\" args=\"[0, 1]\" />\n...\n\nFinal Answer: FALSE",
    "label": "FALSE",
    "num_steps": 5
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `problem_id` | Unique identifier (encodes hop count, case, distractor count) |
| `training_text` | Pre-formatted SFT training string |
| `label` | Ground truth: `TRUE`, `FALSE`, or `UNKNOWN` |
| `num_steps` | Number of reasoning steps in gold trace |

### Dataset Statistics

| Dataset | Split | Problems | Notes |
|---------|-------|----------|-------|
| **PrOntoQA** | Train | 14,346 | Synthetic, controlled complexity |
| **PrOntoQA** | Test | 1,594 | Same distribution |
| **FOLIO** | Train | 1,001 | Human-annotated, natural language |
| **FOLIO** | Val | 203 | Expert FOL translations |

---

## 2. Supervised Fine-Tuning (SFT)

### Purpose

Teach the base LLM the **Thought/Action format** with discrete options.

### What the Model Learns

```
INPUT (Prompt):
Premises:
  [0] Wren is a jompus
  [1] Every jompus is nervous
  [2] Every nervous thing is tired
  ...

Conclusion to evaluate: Wren is tired.

Determine if the conclusion is TRUE, FALSE, or UNKNOWN.

OUTPUT (Completion):
Reasoning:
Thought: Since Wren is a jompus (premise 0) and every jompus is nervous (premise 1), by modus ponens, Wren is nervous.
Action: <Option type="MODUS_PONENS" args="[0, 1]" />
Thought: Since Wren is nervous and every nervous thing is tired (premise 2), Wren is tired.
Action: <Option type="MODUS_PONENS" args="[12, 2]" />
Thought: Therefore, the conclusion "Wren is tired" is TRUE.
Action: <Option type="CONCLUDE" args="[0]" />

Final Answer: TRUE
```

### SFT Training Details

- **Data**: Full PrOntoQA train set (14,346 problems)
- **Format**: Causal LM, predict completion given prompt
- **Epochs**: 3
- **LoRA**: r=64, alpha=128, target q/k/v/o projections

### After SFT

✅ Model generates valid Thought/Action syntax  
✅ Model knows option vocabulary  
⚠️ Model may still make **logical errors** (wrong premise indices, invalid rule applications)

---

## 3. Option Vocabulary

The discrete set of inference rules the model can invoke:

| Option | Symbol | Rule | Example |
|--------|--------|------|---------|
| `MODUS_PONENS` | MP | P, P→Q ⊢ Q | From "A is B" and "All B are C", derive "A is C" |
| `MODUS_TOLLENS` | MT | ¬Q, P→Q ⊢ ¬P | From "A is not C" and "All B are C", derive "A is not B" |
| `UNIV_INSTANTIATION` | UI | ∀x.P(x) ⊢ P(c) | From "All cats are mammals", derive "Fluffy is a mammal" |
| `EXIST_INSTANTIATION` | EI | ∃x.P(x) ⊢ P(sk) | From "Some cat exists", derive "sk is a cat" |
| `AND_INTRO` | AI | P, Q ⊢ P∧Q | Combine two facts |
| `AND_ELIM` | AE | P∧Q ⊢ P | Extract one conjunct |
| `OR_INTRO` | OI | P ⊢ P∨Q | Weaken a fact |
| `DISJUNCTIVE_SYLLOGISM` | DS | P∨Q, ¬P ⊢ Q | Eliminate disjunct |
| `HYPOTHETICAL_SYLLOGISM` | HS | P→Q, Q→R ⊢ P→R | Chain implications |
| `DOUBLE_NEGATION` | DN | ¬¬P ⊢ P | Eliminate double negation |
| `CONCLUDE` | DONE | — | Terminal: output TRUE/FALSE/UNKNOWN |

### Action String Format

```
<Option type="MODUS_PONENS" args="[premise_idx_1, premise_idx_2]" />
```

- `type`: Option name (must match vocabulary)
- `args`: List of integers (indices into current formula set)

---

## 4. Trace Generation (OaK Stage 1)

### Purpose

Generate candidate reasoning traces for each problem, then verify with solver.

### Process

```
For each problem:
    1. Load problem (premises + conclusion)
    2. Generate N traces (samples_per_problem)
    3. For each trace:
        - Parse each step's Action
        - Verify with FOL solver
        - Record solver_valid for each step
    4. Save to traces.jsonl
```

### Output Format (`traces.jsonl`)

```json
{
    "problem_id": "ProntoQA_train_hop-1_case-257",
    "label": "FALSE",
    "final_answer": "FALSE",
    "correct": true,
    "num_steps": 4,
    "steps": [
        {
            "step_idx": 0,
            "thought": "Since Wren is a jompus and every jompus is nervous...",
            "action": "<Option type=\"MODUS_PONENS\" args=\"[0, 8]\" />",
            "option_type": "MODUS_PONENS",
            "option_args": [0, 8],
            "solver_valid": true,
            "solver_error": null
        },
        {
            "step_idx": 1,
            "thought": "Therefore, the conclusion is false.",
            "action": "<Option type=\"CONCLUDE\" args=\"[1]\" />",
            "option_type": "CONCLUDE",
            "option_args": [1],
            "solver_valid": true,
            "solver_error": null
        }
    ],
    "all_steps_valid": true,
    "valid_step_count": 2,
    "total_step_count": 2
}
```

### Key Fields for DPO

| Field | Type | Description |
|-------|------|-------------|
| `solver_valid` | bool | Did this step pass FOL solver verification? |
| `all_steps_valid` | bool | Are ALL steps in this trace valid? |
| `correct` | bool | Does `final_answer` match `label`? |
| `valid_step_count` | int | Number of valid steps |
| `total_step_count` | int | Total steps with solver verdicts |

### Generation Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--temperature` | Sampling temperature | 0.0 (greedy) or 0.7 (diverse) |
| `--samples-per-problem` | Traces per problem | 2-4 |
| `--max-steps` | Maximum reasoning steps | 6-8 |
| `--max-thought-tokens` | Token limit for Thought | 60-100 |
| `--max-action-tokens` | Token limit for Action | 25 |

---

## 5. Preference Pair Construction

### Purpose

Create (chosen, rejected) pairs for DPO training based on solver verification.

### Scoring Logic

```python
def score_trace(trace):
    score = trace["valid_step_count"] / trace["total_step_count"]  # validity rate
    
    if trace["correct"]:
        score += 1.0  # Bonus for correct final answer
    
    if trace["all_steps_valid"]:
        score += 0.5  # Bonus for fully valid trace
    
    return score
```

### Pair Selection

For each problem with multiple traces:

1. **Sort** traces by score (descending)
2. **Chosen**: Highest-scoring trace (ideally: correct + all steps valid)
3. **Rejected**: Lower-scoring trace (wrong answer or invalid steps)
4. **Skip** if no valid contrast (all traces identical scores)

### Output Format (`preference_pairs.jsonl`)

```json
{
    "problem_id": "ProntoQA_train_hop-1_case-257",
    "prompt": "Premises:\n  [0] Wren is a jompus...\nConclusion: ...",
    "chosen": "Thought: Since Wren is a jompus...\nAction: <Option type=\"MODUS_PONENS\" args=\"[0, 8]\" />\n...\nFinal Answer: FALSE",
    "rejected": "Thought: Let me check if Wren is large...\nAction: <Option type=\"MODUS_PONENS\" args=\"[0, 10]\" />\n...\nFinal Answer: TRUE",
    "chosen_score": 2.5,
    "rejected_score": 0.3
}
```

---

## 6. DPO Training (OaK Stage 2)

### Purpose

Align the model to prefer solver-valid traces over invalid ones.

### DPO Objective

```
L_DPO = -E[log σ(β * (log π(chosen|x)/π_ref(chosen|x) - log π(rejected|x)/π_ref(rejected|x)))]
```

Where:
- `π`: Current policy (being trained)
- `π_ref`: Reference policy (SFT model)
- `β`: Temperature parameter (0.1)

### What DPO Teaches

✅ Prefer correct premise indices  
✅ Prefer valid rule applications  
✅ Prefer correct final answers  
❌ Avoid hallucinated facts  
❌ Avoid logical fallacies

### Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--beta` | 0.1 | KL penalty weight |
| `--num-epochs` | 1 | Per iteration |
| `--batch-size` | 2-4 | Per GPU |
| `--learning-rate` | 5e-6 | Lower than SFT |
| `--gradient-accumulation-steps` | 4-8 | For effective batch |

---

## 7. The Full OaK Loop

### Iteration Flow

```
Iteration 0:
  Model: SFT model
  → Generate traces
  → Verify with solver
  → Build preference pairs
  → DPO training
  → Output: DPO model (iter 0)

Iteration 1:
  Model: DPO model (iter 0)
  → Generate traces (should be better!)
  → Verify with solver
  → Build preference pairs
  → DPO training
  → Output: DPO model (iter 1) = FINAL
```

### Expected Improvement

| Metric | SFT | Iter 0 | Iter 1 |
|--------|-----|--------|--------|
| Accuracy | ~70% | ~75% | ~80% |
| Step Validity | ~65% | ~75% | ~85% |
| Trace Validity | ~45% | ~55% | ~65% |

### Why Iteration Helps

1. **Better traces** → More valid/invalid contrast
2. **More preference signal** → Stronger DPO learning
3. **Compounding improvement** → Each iteration builds on the last

---

## 8. Paper vs Implementation

### Notation Differences

| Paper | Implementation | Notes |
|-------|----------------|-------|
| `UNIV_INST(0, Stella)` | `<Option type="UNIV_INSTANTIATION" args="[0, 'Stella']" />` | Paper uses shorthand |
| Algorithm 1 (monolithic) | Two-stage scripts | Same logic, different organization |
| $\hat{q}_\phi$ | Optional `--train-option-head` | Can be skipped for speed |

### What's NOT in the Paper (Implementation Details)

1. **Two-stage separation**: `generate_traces.py` + `train_dpo_from_traces.py`
2. **Multi-GPU parallelization**: `--num-gpus` flag
3. **Scoring formula**: `validity_rate + correct_bonus + valid_trace_bonus`
4. **Greedy vs sampling trade-off**: `--temperature 0.0` vs `0.7`

---

## 9. Quick Reference Commands

```bash
# Stage 1: Generate traces
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/generate_traces.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/iter0 \
    --num-problems 1500 \
    --samples-per-problem 2 \
    --num-gpus 6

# Stage 2: Train DPO
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --num_processes=6 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter0/traces.jsonl \
    --model outputs/sft/latest/final \
    --output outputs/dpo/iter0

# Full loop (automated)
./scripts/run_oak_loop.sh 2 1500 "2,3,4,5,6,7"
```

---

## 10. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| No preference pairs | All traces identical | Increase `--samples-per-problem` or `--temperature` |
| OOM during generation | Model too large | Use `--num-gpus` to distribute |
| Slow trace generation | Sequential processing | Enable multi-GPU with `--num-gpus 6` |
| Low step validity | Model not well-trained | Check SFT quality first |
| DPO not improving | Too few pairs | Need >100 pairs minimum |

