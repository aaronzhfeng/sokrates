# SOKRATES Documentation Index

> **SOKRATES**: **S**ymbolic **O**ption-**K**nowledge **R**easoning **A**lignment via **T**race **E**valuation with **S**olver

A neuro-symbolic approach to logical reasoning that instantiates Sutton's Options and Knowledge (OaK) architecture in a first-order logic micro-world.

---

## Quick Links

| Document | Description |
|----------|-------------|
| [01 - Title & Abstract](01_title_and_abstract.md) | Paper title and abstract |
| [02 - Paper Plan](02_paper_plan.md) | Full paper structure and implementation timeline |
| [03 - Project Plan](03_project_plan.md) | Detailed project design and experimental setup |
| [04 - OaK Connection](04_oak_connection_notes.md) | Analysis of alignment with Sutton's OaK framework |
| [05 - Technical Spec](05_technical_spec.md) | Implementation details, data structures, and code architecture |
| [06 - Glossary](06_glossary.md) | Definitions of key terms and concepts |
| [07 - Implementation Guide](07_implementation_guide.md) | Complete code walkthrough with examples |
| [08 - API Reference](08_api_reference.md) | Full API documentation for all modules |

---

## Project Overview

### The Problem
Large language models produce **logically invalid** chain-of-thought reasoning even when their final answers are correct. Existing approaches treat reasoning as unstructured text and don't model reasoning actions or their reliability.

### Our Solution
SOKRATES represents proofs as sequences of **discrete reasoning options** (inference-rule macros) and uses a **FOL solver** to provide ground-truth knowledge about option validity. This knowledge is distilled into the LLM via **Direct Preference Optimization (DPO)**.

### Key Innovations
1. **Explicit Options** — Finite vocabulary of reusable inference macros
2. **Explicit Knowledge** — Learned option-success predictor q̂_φ(s,ω)
3. **Micro OaK Loop** — Iterative generate → verify → retrain cycle

---

## Architecture Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           SOKRATES Pipeline              │
                    └─────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐            ┌───────────────┐            ┌───────────────┐
│   Datasets    │            │  Base LLM +   │            │  FOL Solver   │
│  P-FOLIO      │            │  Option Head  │            │  (verifier)   │
│  PrOntoQA     │            │  q̂_φ(s,ω)     │            │               │
└───────┬───────┘            └───────┬───────┘            └───────┬───────┘
        │                            │                            │
        │  Optionize                 │  Generate                  │  Verify
        ▼                            ▼                            ▼
┌───────────────┐            ┌───────────────┐            ┌───────────────┐
│   Thought/    │            │  Optionized   │            │ Step Validity │
│   Action      │◀──────────▶│   Traces      │◀──────────▶│   Labels      │
│   Format      │            │               │            │               │
└───────────────┘            └───────┬───────┘            └───────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
            │  Train q̂_φ  │  │   Build     │  │    DPO      │
            │  (knowledge)│  │   Prefs     │  │   (policy)  │
            └─────────────┘  └─────────────┘  └─────────────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │   Repeat    │
                              │  2-3 times  │
                              └─────────────┘
```

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **0: Setup** | 0.5 week | Environment, base model selection |
| **1: Data** | 1 week | Data loaders, optionizer, solver wrapper |
| **2: SFT** | 1 week | Optionized SFT model, constrained decoding |
| **3: OaK-DPO** | 1.5 weeks | Full training loop, preference construction |
| **4: Eval** | 1 week | Metrics, ablations, paper figures |

**Total: ~4 weeks**

---

## Target Venue

**AAAI-26 Bridge Workshop: Logical and Symbolic Reasoning in Language Models**

### Fit with Workshop Topics
- ✓ Using logic to enhance LLM reasoning
- ✓ Avoiding contradictions in generation
- ✓ Symbolic integration with neural models
- ✓ Benchmark evaluation (FOLIO, PrOntoQA)

---

## Quick Start (Future)

```bash
# Clone repository
git clone https://github.com/[user]/sokrates.git
cd sokrates

# Install dependencies
pip install -e .

# Prepare data
python scripts/prepare_data.py

# Run SFT
python scripts/train_sft.py --config configs/sft.yaml

# Run OaK-DPO loop
python scripts/run_oak_dpo.py --config configs/oak_dpo.yaml

# Evaluate
python scripts/evaluate.py --model checkpoints/final --output results/
```

---

## Citation (Draft)

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

