# Ablation Studies

> Document created: December 11, 2025

This document describes the ablation studies conducted to validate key design choices in SOKRATES.

---

## Motivation

SOKRATES has two key components that differentiate it from standard approaches:

1. **Optionized Format** — Representing reasoning steps as discrete actions with explicit rule types and argument references (e.g., `<Option type="MODUS_PONENS" args="[0, 1]" />`)

2. **Solver-in-the-Loop DPO** — Using an FOL solver to verify each reasoning step and construct preference pairs based on step-level validity, not just final answer correctness

To validate that these components contribute to performance, we conduct ablation studies that remove each component individually.

---

## Ablation: w/o Solver Verification (Answer-only DPO)

### Research Question
> Does solver-verified step validity matter for DPO training, or is final answer correctness sufficient?

### Experimental Setup

**Control (Main SOKRATES):**
- DPO preference pairs constructed based on:
  - Trace with MORE valid steps is **chosen**
  - Trace with FEWER valid steps is **rejected**
- Solver verifies each step's logical validity

**Ablation (Answer-only DPO):**
- DPO preference pairs constructed based on:
  - Trace with **correct final answer** is **chosen**
  - Trace with **incorrect final answer** is **rejected**
- Solver verification is **completely ignored** during preference construction

### Implementation

The ablation was implemented in `scripts/train_dpo_answer_only.py`:

```python
def build_preference_pairs_answer_only(traces: list[dict]) -> list[dict]:
    """
    Build DPO preference pairs based ONLY on answer correctness.
    
    This ignores step validity entirely - only the final answer matters.
    This simulates "w/o solver verification" ablation.
    """
    # Group traces by problem
    problem_to_traces = defaultdict(list)
    for trace in traces:
        problem_id = trace["problem_id"]
        problem_to_traces[problem_id].append(trace)
    
    pairs = []
    for problem_id, problem_traces in problem_to_traces.items():
        # Split into correct and incorrect based on ANSWER only
        correct_traces = [t for t in problem_traces if t["correct"]]
        incorrect_traces = [t for t in problem_traces if not t["correct"]]
        
        # Need at least one of each
        if not correct_traces or not incorrect_traces:
            continue
        
        # Create pairs: correct answer vs incorrect answer
        for correct_trace in correct_traces[:2]:
            for incorrect_trace in incorrect_traces[:2]:
                pairs.append({
                    "prompt": build_prompt(correct_trace),
                    "chosen": build_trace_text(correct_trace),
                    "rejected": build_trace_text(incorrect_trace),
                    # Note: we ignore step validity here
                })
    
    return pairs
```

### Results

| Model | Test Accuracy | Valid Traces | Step Validity |
|-------|---------------|--------------|---------------|
| **SOKRATES DPO iter1** | ~99% | High | High |
| **w/o solver (Answer-only DPO)** | **95.5%** | ~2.2% | ~31.6% |

### Key Findings

1. **Answer accuracy remains high** — Even without solver verification, the model achieves 95.5% accuracy on PrOntoQA test set.

2. **Step validity drops dramatically** — Only ~2.2% of traces are fully valid (all steps verified correct), compared to much higher rates with solver verification.

3. **The gap is meaningful** — The ~3.5% accuracy difference plus the large step validity gap demonstrates that **solver-in-the-loop DPO improves both final answers AND reasoning quality**.

### Interpretation

Without solver verification, DPO only teaches the model to produce outputs that lead to correct answers. The model may:
- Take "shortcuts" that skip valid reasoning steps
- Produce logically inconsistent intermediate steps that happen to reach the right answer
- Not learn which inference rules are valid in which contexts

With solver verification, DPO teaches the model to produce **verifiably correct reasoning chains**, not just correct answers. This is crucial for:
- Trustworthy AI reasoning
- Generalization to harder problems
- Interpretable decision-making

---

## Summary Table

| Ablation | What's Removed | Accuracy | Step Validity | Conclusion |
|----------|----------------|----------|---------------|------------|
| Full SOKRATES | Nothing | ~99% | High | Baseline |
| w/o Solver | Step-level verification | 95.5% | ~2% | Solver verification crucial for reasoning quality |

---

## Implications for Paper

These ablation results support the core claim of SOKRATES:

> **Solver-in-the-loop DPO produces models that are not only more accurate, but also produce verifiably correct reasoning chains.**

The ~3.5% accuracy improvement is meaningful, but the real value is in the dramatically higher step validity — demonstrating that SOKRATES learns to reason correctly, not just to produce correct answers through potentially invalid shortcuts.

---

## Files & Artifacts

| Artifact | Location |
|----------|----------|
| Answer-only DPO script | `scripts/train_dpo_answer_only.py` |
| Ablation shell script | `scripts/ablation_3_answer_only_dpo.sh` |
| Trained model | `outputs/ablation/answer_only_dpo/dpo/merged` |
| Evaluation results | `outputs/ablation/answer_only_dpo/eval/` |
| Summary JSON | `outputs/ablation/answer_only_dpo/summary.json` |

