# SOKRATES Experimental Approach & Key Decisions

> **Purpose**: This document provides context for agents and collaborators about the experimental design, key decisions, and current state of the SOKRATES project.

---

## 1. Executive Summary

SOKRATES implements an **Options and Knowledge (OaK)** approach to improve LLM logical reasoning. The key insight is:

> **Train on synthetic data (PrOntoQA), transfer to real-world data (FOLIO)**

We use PrOntoQA for SFT because it provides:
- Clean, unambiguous logical structure
- Ground-truth reasoning chains
- Unlimited synthetic examples

Then we test transfer to FOLIO (human-curated Wikipedia examples) to validate generalization.

---

## 2. Two-Dataset Strategy

### 2.1 PrOntoQA (Primary Training Dataset)

| Property | Value |
|----------|-------|
| Source | Synthetic generation |
| Size | 14,346 train / 1,594 test |
| Logic Type | Syllogistic (simple FOL subset) |
| Proof Chains | Ground-truth provided |
| Ambiguity | None |

**Why PrOntoQA for SFT:**
1. **Clean supervision**: Every example has a correct, unambiguous proof chain
2. **Controllable complexity**: Can generate arbitrary hop-lengths
3. **No annotation noise**: Synthetic = perfect labels
4. **Scalable**: Can generate more data if needed

### 2.2 FOLIO (Transfer/Evaluation Dataset)

| Property | Value |
|----------|-------|
| Source | Human-curated from Wikipedia |
| Size | 1,001 train / 203 validation |
| Logic Type | Full first-order logic |
| Proof Chains | Not provided (only conclusions) |
| Ambiguity | Some examples have debatable labels |

**Why NOT SFT on FOLIO:**
1. **No proof chains**: FOLIO only provides premises + conclusion, not reasoning steps
2. **Label noise**: Some examples have controversial annotations
3. **Small size**: 1,001 examples is insufficient for format learning
4. **Complex FOL**: Requires Z3 solver, harder to verify individual steps

---

## 3. Training Pipeline Design

### 3.1 Phase 1: SFT on PrOntoQA Only

```
┌─────────────────────────────────────────────────────────────────┐
│  SFT Training (PrOntoQA)                                        │
├─────────────────────────────────────────────────────────────────┤
│  Input:  14,346 PrOntoQA examples with optionized traces        │
│  Model:  Qwen/Qwen3-8B + LoRA (r=64, α=128)                     │
│  Output: Model that generates Thought/Action format             │
│                                                                 │
│  Key: Model learns the OUTPUT FORMAT from clean synthetic data  │
└─────────────────────────────────────────────────────────────────┘
```

**What the model learns in SFT:**
- The Thought/Action output structure
- How to reference premises by index `[0]`, `[1]`, etc.
- The Option types: `MODUS_PONENS`, `MODUS_TOLLENS`, `CONCLUDE`
- When to conclude TRUE/FALSE/UNKNOWN

### 3.2 Phase 2: OaK-DPO on PrOntoQA

```
┌─────────────────────────────────────────────────────────────────┐
│  OaK-DPO Loop (PrOntoQA) - 3 Iterations                         │
├─────────────────────────────────────────────────────────────────┤
│  For each iteration:                                            │
│    1. Generate 2 traces per problem (temp=0.5)                  │
│    2. Verify with PrOntoQA solver                               │
│    3. Build preference pairs (valid > invalid)                  │
│    4. Train DPO (β=0.1, lr=5e-7)                                │
│                                                                 │
│  Key: Model learns to PREFER valid reasoning from solver signal │
└─────────────────────────────────────────────────────────────────┘
```

**What the model learns in DPO:**
- Which reasoning steps are logically valid
- How to correctly chain premises together
- To avoid invalid argument indices

### 3.3 Phase 3: Transfer to FOLIO (Future Work)

```
┌─────────────────────────────────────────────────────────────────┐
│  FOLIO Evaluation/Training                                      │
├─────────────────────────────────────────────────────────────────┤
│  Option A: Zero-shot transfer                                   │
│    - Apply PrOntoQA-trained model directly to FOLIO             │
│    - Tests generalization of optionized reasoning               │
│                                                                 │
│  Option B: FOLIO DPO (no SFT)                                   │
│    - Generate traces on FOLIO                                   │
│    - Verify with Z3 solver                                      │
│    - DPO training (builds on PrOntoQA foundation)               │
│                                                                 │
│  Key: Skip SFT on FOLIO because format is already learned       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Experimental Decisions

### 4.1 Decision: No SFT on FOLIO

**Rationale:**
| Factor | PrOntoQA | FOLIO |
|--------|----------|-------|
| Has proof chains | ✅ Yes | ❌ No |
| Clean labels | ✅ Perfect | ⚠️ Some noise |
| Sufficient size | ✅ 14K examples | ❌ 1K examples |
| Solver verification | ✅ Pattern-based | ⚠️ Z3 (complex) |

**The core insight**: SFT teaches OUTPUT FORMAT, not domain knowledge. Once the model knows how to produce `Thought: ... Action: <Option .../>`, it can apply this format to any domain. DPO then teaches domain-specific validity.

### 4.2 Decision: Subset for DPO (10% of training data)

**Rationale:**
- DPO is sample-efficient (converges with ~1-5K preference pairs)
- Trace generation is compute-intensive (N problems × K samples × S steps)
- Literature shows diminishing returns beyond ~3K pairs

**Implementation:**
- DPO uses 1,500 problems × 2 samples = 3,000 traces per iteration
- Full test set (1,594) always used for evaluation

### 4.3 Decision: 3 DPO Iterations

**Rationale:**
- Iteration 1: Big jump in validity (11% → 45%)
- Iteration 2: Continued improvement (45% → 84%)
- Iteration 3: Near convergence (84% → 92%)

**Observation:** Accuracy plateaus after iter2, but step validity keeps improving. This validates the OaK approach - the model learns *better reasoning*, not just *correct answers*.

### 4.4 Decision: Temperature 0.5 for DPO Generation

**Rationale:**
- Temperature 0.0: No diversity → bad for DPO (needs both valid/invalid traces)
- Temperature 1.0: Too random → mostly invalid traces
- Temperature 0.5: Good balance of quality and diversity

**Critical Warning:** Using temperature=0.0 for DPO trace generation eliminates diversity and cripples DPO training!

---

## 5. Optionized Format

### 5.1 Input Format (Problem Representation)

```
Premises:
  [0] Wren is a jompus.
  [1] Every jompus is small.
  [2] Every jompus is a wumpus.
  [3] Every wumpus is opaque.
  [4] Wumpuses are not floral.

Conclusion to evaluate: Wren is opaque.

Determine if the conclusion is TRUE, FALSE, or UNKNOWN.

Reasoning:
```

### 5.2 Output Format (Reasoning Trace)

```
Thought: Since Wren is a jompus (premise 0) and every jompus is a wumpus (premise 2), 
we can derive that Wren is a wumpus.
Action: <Option type="MODUS_PONENS" args="[0, 2]" />
Thought: Now that Wren is a wumpus (derived as premise 5) and every wumpus is opaque (premise 3), 
we can conclude Wren is opaque.
Action: <Option type="MODUS_PONENS" args="[5, 3]" />
Thought: This matches our target conclusion exactly, so the answer is TRUE.
Action: <Option type="CONCLUDE" args="[0]" />
```

### 5.3 Option Types

| Option | Args | Meaning |
|--------|------|---------|
| `MODUS_PONENS` | `[fact_idx, rule_idx]` | Apply P + (P→Q) = Q |
| `MODUS_TOLLENS` | `[neg_q_idx, rule_idx]` | Apply ¬Q + (P→Q) = ¬P |
| `CONCLUDE` | `[0]`, `[1]`, or `[2]` | TRUE, FALSE, UNKNOWN |

---

## 6. Current Results (PrOntoQA)

### 6.1 Accuracy Progression

| Model | Accuracy | Step Validity | Valid Traces |
|-------|----------|---------------|--------------|
| Base Qwen3-8B | ~85% | - | - |
| + SFT | 93.3% | 11.3% | 3.2% |
| + DPO iter1 | 96.8% | 44.7% | 28.5% |
| + DPO iter2 | 98.1% | 83.5% | 71.2% |
| + DPO iter3 | **98.2%** | **91.8%** | **85.4%** |

### 6.2 Key Observations

1. **Accuracy vs Validity Gap**: SFT achieves 93% accuracy but only 11% valid steps → the model gets correct answers through shortcuts
2. **OaK Closes the Gap**: DPO iterations push validity up while maintaining accuracy
3. **Diminishing Returns**: Accuracy plateaus at iter2, validity continues improving through iter3

---

## 7. Artifacts and Storage

### 7.1 Models (HuggingFace)

| Model | HuggingFace Path |
|-------|------------------|
| SFT | `Moonlight556/sokrates-qwen3-8b-prontoqa-sft-optionized` |
| DPO iter1 | `Moonlight556/sokrates-qwen3-8b-prontoqa-oak-dpo-iter1` |
| DPO iter2 | `Moonlight556/sokrates-qwen3-8b-prontoqa-oak-dpo-iter2` |
| DPO iter3 | `Moonlight556/sokrates-qwen3-8b-prontoqa-oak-dpo-iter3` |

### 7.2 Dataset (HuggingFace)

**Repository:** `Moonlight556/sokrates-prontoqa-data`

| Path | Description |
|------|-------------|
| `processed/prontoqa_train.jsonl` | SFT training data |
| `processed/prontoqa_test.jsonl` | Test data |
| `processed/folio_*.jsonl` | FOLIO data (for future use) |
| `traces/iter0_traces.jsonl` | Traces from SFT model |
| `traces/iter1_traces.jsonl` | Traces from DPO iter1 |
| `traces/iter2_traces.jsonl` | Traces from DPO iter2 |
| `eval/prontoqa_final/` | Final evaluation traces |

### 7.3 Local Artifacts (results/)

```
results/
├── sft/
│   ├── summary.json           # SFT training metrics
│   └── training_history.json  # Loss curves
├── dpo/
│   ├── iter1/                 # DPO iteration 1
│   ├── iter2/                 # DPO iteration 2
│   └── iter3/                 # DPO iteration 3
├── traces/
│   ├── iter0/                 # Generated traces
│   ├── iter1/
│   └── iter2/
├── eval/
│   └── prontoqa_final/        # Final evaluation
└── analysis/
    └── data_analysis.json     # Data statistics
```

---

## 8. Scripts Reference

### 8.1 Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/prepare_data.py` | Download and optionize datasets |
| `scripts/train_sft.py` | Run SFT training |
| `scripts/train_dpo.py` | Run DPO training |
| `scripts/generate_traces_vllm.py` | Generate traces with vLLM |
| `scripts/merge_lora_adapter.py` | Merge LoRA into base model |

### 8.2 Automation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_vllm_parallel.sh` | Parallel trace generation (6 GPUs) |
| `scripts/run_prontoqa_remaining.sh` | Complete PrOntoQA pipeline |
| `scripts/run_folio_full.sh` | Full FOLIO pipeline |

---

## 9. Common Questions (FAQ)

### Q: Why not fine-tune on FOLIO?

**A:** FOLIO doesn't provide reasoning chains, only (premises, conclusion, label). We can't do SFT without supervision on the *reasoning process*. The model would only learn to predict TRUE/FALSE/UNKNOWN without learning *why*.

### Q: Will the PrOntoQA model work on FOLIO?

**A:** Partially. The model knows the output format and basic inference rules. However:
- FOLIO uses more complex FOL (quantifiers, negation, etc.)
- FOLIO premises are real-world text (less structured)
- We expect ~70-80% transfer without FOLIO-specific DPO

### Q: Why use Qwen3-8B instead of Llama?

**A:** Qwen3-8B showed better out-of-box logical reasoning in preliminary tests. Also, Qwen's tokenizer handles the XML-like Option format slightly better.

### Q: Why vLLM instead of HuggingFace generate()?

**A:** Speed. HuggingFace generate() was processing ~50 problems/hour. vLLM with 6× data parallelism processes ~2,000 problems/hour (40× speedup).

### Q: What if I want to run on fewer GPUs?

**A:** Adjust `run_vllm_parallel.sh`:
- Change `GPUS="2 3 4 5 6 7"` to your available GPUs
- Each GPU processes an equal share of problems
- Minimum: 1 GPU works but is slower

---

## 10. Future Work

### 10.1 Immediate Next Steps

1. **FOLIO evaluation**: Test PrOntoQA model on FOLIO validation set
2. **FOLIO DPO**: Run OaK-DPO on FOLIO using Z3 solver
3. **Ablation studies**: Impact of iterations, samples, temperature

### 10.2 Potential Improvements

- [ ] Multi-task training: Joint PrOntoQA + FOLIO
- [ ] Curriculum learning: Easy → hard problems
- [ ] Option expansion: Add more inference rules
- [ ] Knowledge head: Train explicit q̂_φ predictor

---

## 11. Contact and Resources

- **GitHub**: [Repository URL]
- **HuggingFace**: https://huggingface.co/Moonlight556
- **Documentation**: This `docs/` folder
- **Paper Draft**: `paper/` folder

---

*Last updated: December 11, 2025*

