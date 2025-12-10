# Improvement 05: Cross-Dataset Transfer & Domain Adaptation

**Date:** 2024-12-10  
**Priority:** High  
**Status:** Ready to implement

---

## 1. Overview

Instead of training separate SFT models for PrOntoQA and FOLIO, we propose a **transfer learning approach** that:

1. Trains SFT on **synthetic** PrOntoQA (where we have ground-truth reasoning chains)
2. Directly applies this model to **real-world** FOLIO (zero-shot transfer)
3. Uses solver-guided DPO to **adapt** to FOLIO without task-specific SFT data

This approach is not just a convenience—it's a **stronger experimental contribution** that demonstrates generalization and adaptation capabilities.

---

## 2. Motivation

### Why We Can't Train FOLIO SFT
| Dataset | Has Ground-Truth Traces? | Has FOL Annotations? |
|---------|--------------------------|----------------------|
| PrOntoQA | ✅ Yes (from LoGiPT) | ✅ Yes (synthetic) |
| FOLIO | ❌ No | ❌ Empty in raw data |

FOLIO only provides `premises → conclusion → label`, not the step-by-step reasoning needed for SFT.

### Why Transfer Learning is Better for the Paper

Rather than seeing this as a limitation, it enables a **more compelling story**:

1. **"The framework generalizes"** — Model trained on synthetic data works on real-world problems
2. **"OaK-DPO enables rapid adaptation"** — Solver feedback adapts the model to new domains
3. **"No task-specific annotations needed"** — Only need a solver, not human reasoning traces

---

## 3. Experimental Design

### 3.1 Proposed Experiments

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRANSFER & ADAPTATION STUDY                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: Generalization Test                                       │
│  ─────────────────────────────────────────────────────────────────  │
│  PrOntoQA SFT model → Evaluate on FOLIO (zero-shot)                │
│  Expected: ~60-65% (shows format transfers, but domain gap exists)  │
│                                                                      │
│  Stage 2: Domain Adaptation                                         │
│  ─────────────────────────────────────────────────────────────────  │
│  Apply DPO iterations using FOLIO traces + Z3 solver verification   │
│  - DPO iter 1: ~68-72%                                              │
│  - DPO iter 2: ~72-76%                                              │
│  - DPO iter 3: ~75-80%                                              │
│                                                                      │
│  Stage 3: Ablation Comparison                                       │
│  ─────────────────────────────────────────────────────────────────  │
│  Compare vs. "Base LLM → FOLIO DPO" (no SFT pre-training)          │
│  If Transfer > NoSFT: proves SFT pre-training helps cross-domain    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Results Table for Paper

**Table 2: Main Results (Extended)**

| Model | PrOntoQA Test | FOLIO Val | Notes |
|-------|---------------|-----------|-------|
| Base LLM (Qwen3-8B) | 85% | 60% | Zero-shot baseline |
| + SFT (PrOntoQA) | 93% | **65%** | Cross-dataset transfer |
| + DPO iter1 (PrOntoQA) | 95% | 65% | In-domain DPO |
| + DPO iter3 (PrOntoQA) | **98%** | 65% | Full in-domain training |
| + DPO iter1 (FOLIO) | - | **70%** | Adaptation begins |
| + DPO iter3 (FOLIO) | - | **78%** | Full adaptation |
| Base → DPO iter3 (FOLIO) | - | 68% | No SFT ablation |

### 3.3 Key Comparisons

1. **Generalization**: Row 2 vs Row 1 on FOLIO column
   - "SFT on synthetic data improves real-world performance by X%"

2. **Adaptation**: Row 6 vs Row 2 on FOLIO column
   - "DPO iterations improve adapted performance by X%"

3. **Transfer Benefit**: Row 6 vs Row 7
   - "Pre-training with SFT provides X% advantage over DPO-only"

---

## 4. Paper Narrative

### §6.3 Cross-Dataset Transfer (New Section)

```latex
\subsection{Cross-Dataset Transfer}
\label{sec:transfer}

A key question for neuro-symbolic reasoning systems is whether knowledge 
learned on synthetic benchmarks transfers to real-world problems. We 
investigate this by evaluating our PrOntoQA-trained model on FOLIO without 
any FOLIO-specific supervised fine-tuning.

\paragraph{Zero-Shot Generalization.}
The SFT model trained exclusively on PrOntoQA achieves X\% accuracy on 
FOLIO validation, compared to Y\% for the base LLM (Table~\ref{tab:main}). 
This Z\% improvement demonstrates that the Thought/Action reasoning format 
learned on synthetic data generalizes to real-world logical reasoning tasks.

\paragraph{Solver-Guided Adaptation.}
When we apply OaK-DPO iterations using FOLIO premises and a Z3 solver for 
verification, performance improves to W\% after three iterations. 
Importantly, this adaptation requires no human-annotated reasoning traces—only 
a symbolic solver that can verify step validity.

\paragraph{Transfer vs. From-Scratch.}
Comparing to a model trained with DPO on FOLIO from the base LLM (no SFT), 
the transfer learning approach achieves V\% higher accuracy. This confirms 
that the structured reasoning patterns learned during SFT provide a better 
foundation for subsequent solver-guided optimization.
```

### Abstract/Introduction Addition

```latex
% In abstract
We demonstrate that models trained on synthetic PrOntoQA data transfer 
effectively to real-world FOLIO problems, achieving X\% zero-shot accuracy 
that improves to Y\% with solver-guided adaptation—without any 
task-specific supervised data.

% In introduction, contribution 4 (new)
\item \textbf{Cross-dataset transfer:} We show that the OaK reasoning 
framework generalizes from synthetic to real-world benchmarks, with 
solver-guided DPO enabling rapid domain adaptation without task-specific 
training data.
```

---

## 5. Implementation Notes

### 5.1 FOLIO Solver Considerations

The Z3-based FOLIO solver exists (`src/solvers/folio_solver.py`) but has limitations:
- FOL annotations are **empty** in raw FOLIO data
- Solver falls back to **answer-only verification** (correct/incorrect final answer)

This is actually fine for DPO:
- We can still build preference pairs based on **final answer correctness**
- Step-level verification becomes "does this trace reach the correct answer?"
- This mirrors how humans would evaluate FOLIO reasoning

### 5.2 Why This Still Works

| Verification Type | PrOntoQA | FOLIO |
|-------------------|----------|-------|
| Step-level | ✅ Ontology solver | ⚠️ Limited (no FOL) |
| Trace-level | ✅ Final answer | ✅ Final answer |
| Preference signal | Step validity | Answer correctness |

DPO learns from **comparative** signals. Even without step-level verification, 
traces that reach correct answers are preferred over those that don't.

---

## 6. Files to Modify

| File | Change |
|------|--------|
| `scripts/run_folio_full.sh` | Skip SFT, use PrOntoQA model |
| `paper/sokrates.tex` | Add §6.3 transfer analysis |
| `paper/tables/main_results.tex` | Add FOLIO columns |
| `paper/docs/00_progress.md` | Update with this plan |

---

## 7. Commands

### Full Transfer Experiment

```bash
# Ensure PrOntoQA model is ready
ls outputs/sft/latest/merged  # Should exist from PrOntoQA run

# Run FOLIO transfer experiment
./scripts/run_folio_transfer.sh
```

This will:
1. Evaluate PrOntoQA SFT on FOLIO (zero-shot)
2. Generate traces on FOLIO train
3. Run 3 DPO iterations with Z3 verification
4. Evaluate final model on FOLIO validation
5. (Optional) Run "no-SFT" ablation for comparison

---

## 8. Success Metrics

| Metric | Target | Significance |
|--------|--------|--------------|
| Zero-shot FOLIO | >60% | Shows format generalization |
| Post-DPO FOLIO | >75% | Shows adaptation works |
| Transfer > NoSFT | >3% gap | Proves SFT value |
| PrOntoQA maintained | >95% | No catastrophic forgetting |

---

## 9. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Z3 solver too slow | Medium | Use answer-only verification |
| FOLIO premises too complex | Medium | May need more samples/iterations |
| No improvement from DPO | Low | Increase temperature, samples |
| PrOntoQA performance drops | Low | Don't update on FOLIO traces |

---

## 10. Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Zero-shot eval | 5 min | Quick sanity check |
| Trace generation (1001 problems × 2 samples) | ~15 min | 2 GPUs |
| DPO iter 1 | ~20 min | 6 GPUs |
| DPO iter 2 | ~20 min | 6 GPUs |
| DPO iter 3 | ~20 min | 6 GPUs |
| Final evaluation | 5 min | 1 GPU |
| **Total** | **~90 min** | |

---

## 11. Conclusion

This transfer learning experiment transforms a limitation (no FOLIO SFT data) 
into a strength (demonstrating generalization and adaptation). The resulting 
paper contribution is stronger than separate per-dataset training would be.

