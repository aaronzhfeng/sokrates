# Future Experiments Plan (Not Yet Run)

**Date:** 2025-12-12  
**Status:** Planning only (no new training/eval runs executed as part of paper-polish)

This document records **high-ROI experimental extensions** identified during the final Stage-3 paper review. The goal is to make these ideas actionable later without disrupting the current submission timeline.

---

## Context: what is already “locked in”

- Current submission focuses on the **two-stage pipeline**:
  1) `scripts/generate_traces.py` (generate + step-verify traces)  
  2) `scripts/train_dpo_from_traces.py` (build prefs + DPO)  
- We are **not running additional experiments** right now. This file is a plan only.

---

## P0: Use \(\hat q_\phi\) at inference time (easy, high payoff)

### Idea A — Best-of-\(K\) reranking by step validity score

- **Method**: sample \(K\) traces, compute a trace score
  \[
  S(\tau)=\prod_t \hat q_\phi(s_{t-1},\omega_t)
  \]
  select the trace with the highest \(S(\tau)\), and output its \texttt{CONCLUDE}.

- **Why it helps**: uses calibration to prefer traces that are consistently “likely-valid”, not just high-confidence final answers.

### Implementation sketch

- **Where**:
  - `src/inference/generate_trace.py`: expose optional reranking over multiple traces.
  - `scripts/evaluate.py`: add flag(s) to enable reranking during evaluation.
- **What’s missing today**: generation currently may not populate `step.predicted_valid`; you’ll need to compute \(\hat q_\phi\) during generation (or post-hoc).

### Minimal evaluation protocol (manual, later)

- Evaluate on PrOntoQA test and report:
  - accuracy, step validity, trace validity
  - solver calls (approx. proportional to total steps verified)

---

## P1: Validity-gated action selection (still simple, stronger than reranking)

### Idea B — “sample M actions → pick max \(\hat q_\phi\)” at each step

- **Method**: at each step \(t\):
  - sample \(M\) candidate Actions (syntax-valid by grammar)
  - score each by \(\hat q_\phi(s_{t-1},\omega)\)
  - execute the argmax action

- **Why it helps**: reduces invalid steps before they happen; may increase trace validity without changing the training recipe.

### Implementation sketch

- **Where**:
  - `src/inference/generate_trace.py` (or the `TraceGenerator` class)
- **Key decisions**:
  - whether to resample Thought each time or keep Thought fixed
  - whether to cap backtracking depth (optional)

---

## P2: Preference-design upgrades (paper-reviewer magnet)

### Idea C — Lexicographic preferences (validity-first)

Replace the additive score with an explicit priority:

1) prefer fully valid traces over any invalid trace  
2) among fully valid, prefer correct answer  
3) among ties, prefer higher step-validity / shorter traces

### Idea D — Ablate the “fully-valid bonus” weight

In `scripts/train_dpo_from_traces.py`, the current score includes:

- +1.0 for correct final answer  
- +0.5 for fully valid trace

Add a small sweep over the fully-valid bonus (e.g., 0, 0.5, 1.0, 2.0) to show robustness.

### Implementation sketch

- **Where**: `scripts/train_dpo_from_traces.py`
- **How**:
  - add flags like:
    - `--preference-mode {additive,lexicographic}`
    - `--fullvalid-bonus <float>`
  - keep defaults matching current behavior for backwards compatibility.

---

## P3: Additional ablations / baselines (optional but valuable)

- **Ablation: no Thoughts**
  - Train and/or evaluate with Action-only traces to show “the framework is about actions, not CoT”.
- **Ablation: no constrained decoding**
  - Remove grammar constraint to quantify how much syntactic constraints matter.
- **Ablation: DPO without \(\hat q_\phi\) auxiliary head**
  - Keep solver preferences but remove the predictive head training.
- **Baseline: verifier-based DPO on unstructured CoT**
  - Proxy for VeriCoT-style training: verify free-form CoT with a parser/checker and train DPO on those preferences.

---

## Suggested “minimal compute” schedule (manual, later)

If you want a quick sanity-check run without committing to full-scale experiments:

- **Subset**: 2k PrOntoQA training problems
- **Sampling**: \(K=4\) (to produce more preference diversity)
- **Iterations**: 1
- **GPUs**: prefer `CUDA_VISIBLE_DEVICES=2,3,4,5` (GPU 0 is often busy)

See `docs/13_two_stage_pipeline.md` for the baseline commands; use the same scripts with the new flags once implemented.

---

## Notes (submission constraints)

- **Do not auto-run training**: this repo’s process expects the user to trigger training when ready.
- **Keep paper claims consistent**: if you run any of the above, update the paper tables/claims accordingly (accuracy, step validity, trace validity, solver cost).


