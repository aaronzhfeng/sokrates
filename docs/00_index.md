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
| [09 - Session Log](09_session_log.md) | Development progress, changes, and lessons learned |
| [10 - Experimental Design](10_experimental_design.md) | Data splits, methodology, and rationale |
| [11 - DPO Runtime Notes](11_dpo_runtime_notes.md) | Hotfixes for distributed DPO training |
| [12 - Smoke Test](12_smoke_test.md) | Quick end-to-end validation commands |
| [13 - Two-Stage Pipeline](13_two_stage_pipeline.md) | Decoupled trace generation + DPO training |
| [14 - Debugging Session](14_debugging_session_dec10.md) | Critical bug fixes (Dec 10, 2025) |
| [15 - Optimization Plan](15_optimization_plan.md) | Performance optimization implementation roadmap |
| [16 - Prompt Evolution](16_prompt_evolution.md) | Trace generation prompt design history |
| [17 - vLLM Inference Config](17_vllm_inference_config.md) | vLLM configuration testing & best practices |
| [18 - Hyperparameter Search](18_hyperparameter_search.md) | **NEW:** Temperature, max_steps, samples testing data |

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

## Quick Start

```bash
# Clone repository
git clone https://github.com/sokrates-project/sokrates.git
cd sokrates

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Prepare data (downloads FOLIO and PrOntoQA)
python scripts/prepare_data.py --raw-dir data/raw --output-dir data/processed

# Run SFT training
python scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl

# Run OaK-DPO loop (after SFT)
python scripts/run_oak_dpo.py \
    --config configs/training.yaml \
    --sft-model outputs/sft/latest/final \
    --train-data data/processed/prontoqa_train.jsonl \
    --iterations 3

# Evaluate
python scripts/evaluate.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_test.jsonl \
    --dataset-type prontoqa
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

