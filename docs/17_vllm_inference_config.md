# vLLM Inference Configuration Testing

**Date**: December 10, 2025  
**SFT Model**: `outputs/sft/20251210_041849` (merged to `outputs/sft/latest/merged`)

## Executive Summary

Tested both **throughput** (GPU configs) and **quality** (temperature, max_steps) parameters. 

**Best config for DPO**: Temperature 0.5, max_steps=10-15, 2-4 samples/problem

---

## 1. Quality Testing (Generation Parameters)

### Temperature Effects

| Temp | Accuracy | Step Validity | Valid Traces | **Diversity** |
|------|----------|---------------|--------------|---------------|
| 0.0 (greedy) | 95% | 27.2% | 2% | **6%** ❌ |
| 0.3 | 96% | 27.2% | 6% | 58% |
| **0.5** | **95%** | **36.2%** | **11%** | **78%** ✅ |
| 0.7 | 87% | 35.4% | 6% | 82% |
| 1.0 | 80% | 50.1% | 10% | 86% |

**Key insight**: Temperature 0.5 balances accuracy (95%) with diversity (78%).

### Max Steps Effects (at temp=0.5)

| Max Steps | Accuracy | Step Validity | Valid Traces |
|-----------|----------|---------------|--------------|
| 5 (short) | 90% | 34.3% | **13%** |
| 10 (default) | 95% | 36.2% | 11% |
| 15 (long) | 93% | **41.0%** | 11% |

**Key insight**: Longer traces (15 steps) improve step validity to 41%.

### Samples Per Problem

| Samples | Total Traces | Accuracy | Valid Traces |
|---------|--------------|----------|--------------|
| 2 | 100 | 95% | 11% |
| 4 | 200 | 96% | 9% |

**Key insight**: More samples = more DPO pairs, but diminishing returns.

### Metrics Explained

- **Accuracy**: Final answer (TRUE/FALSE/UNKNOWN) matches ground truth
- **Step Validity**: % of individual reasoning steps that are solver-verified correct
- **Valid Traces**: % of traces where ALL steps are valid
- **Diversity**: % of problems where samples differ (needed for DPO preference pairs!)

---

## 2. Throughput Testing (Hardware Configs)

| Configuration | Throughput | Notes |
|--------------|------------|-------|
| **1 GPU (baseline)** | **3.46 tr/s** | ✅ Best per-GPU efficiency |
| 2 GPU tensor parallel | 2.49 tr/s | ❌ Communication overhead |
| 4 GPU tensor parallel | 2.29 tr/s | ❌ Even worse |
| 6 GPU data parallel | ~20 tr/s | ✅ Linear scaling |

**Winner**: Data parallelism (6 independent processes)

---

## 3. Recommended Configuration

### For DPO Training (need diversity + quality)

```yaml
temperature: 0.5      # Balance accuracy/diversity
max_steps: 15         # Allow multi-step reasoning
samples_per_problem: 2-4  # 2 minimum, 4 for more pairs
```

### Command (6 GPUs)

```bash
./scripts/run_vllm_parallel.sh outputs/traces/iter0 14346
```

### Estimated Time

- 14,346 problems × 2 samples = 28,692 traces
- 6 GPUs × 3.46 tr/s = ~20 traces/sec
- **Total: ~25 minutes**

---

## 4. Why These Settings?

| Setting | Value | Rationale |
|---------|-------|-----------|
| **temp=0.5** | 0.5 | 95% accuracy + 78% diversity (sweet spot) |
| **max_steps=15** | 15 | 41% step validity (higher than 10) |
| **samples=2** | 2 | Minimum for preference pairs |

### DPO Needs

1. **Diverse samples** - different traces for same problem to form (chosen, rejected) pairs
2. **Mixed validity** - some valid, some invalid traces for preference learning
3. **Reasonable accuracy** - most traces should reach correct conclusions

Temperature 0.5 achieves all three.

---

## 5. Session Progress

### Completed
- SFT training: `outputs/sft/20251210_041849` (51 min, 6 GPUs)
- LoRA merge for vLLM: `outputs/sft/latest/merged`
- Quality parameter testing: temp, max_steps, samples
- Throughput testing: tensor vs data parallel

### Findings
- **Greedy (temp=0) is bad**: Only 6% diversity, useless for DPO
- **Tensor parallel hurts**: 8B model too small, communication dominates
- **Data parallel scales**: 6 GPUs = 6× throughput
