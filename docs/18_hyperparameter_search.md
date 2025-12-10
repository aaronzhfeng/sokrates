# Hyperparameter Search: Trace Generation Quality

**Date**: December 10, 2025  
**Model**: Qwen3-8B SFT (`outputs/sft/20251210_041849/merged`)  
**Dataset**: PrOntoQA test set (50 problems per test)  
**Hardware**: NVIDIA B200 (single GPU per test)

## Raw Data

### Temperature Sweep (max_steps=10, samples=2)

| Temperature | Accuracy | Step Validity | Valid Traces | Diversity | Avg Steps | Total Steps | Valid Steps |
|-------------|----------|---------------|--------------|-----------|-----------|-------------|-------------|
| 0.0 | 95.0% | 27.2% | 2.0% | 6.0% | 51.61 | 5161 | 1405 |
| 0.3 | 96.0% | 27.2% | 6.0% | 58.0% | 48.57 | 4857 | 1322 |
| **0.5** | **95.0%** | **36.2%** | **11.0%** | **78.0%** | 48.75 | 4875 | 1764 |
| 0.7 | 87.0% | 35.4% | 6.0% | 82.0% | 45.74 | 4574 | 1617 |
| 1.0 | 80.0% | 50.1% | 10.0% | 86.0% | 37.63 | 3763 | 1884 |

### Max Steps Sweep (temp=0.5, samples=2)

| Max Steps | Accuracy | Step Validity | Valid Traces | Diversity | Avg Steps | Total Steps | Valid Steps |
|-----------|----------|---------------|--------------|-----------|-----------|-------------|-------------|
| 5 | 90.0% | 34.3% | 13.0% | 74.0% | 26.54 | 2654 | 909 |
| 10 | 95.0% | 36.2% | 11.0% | 78.0% | 48.75 | 4875 | 1764 |
| **15** | **93.0%** | **41.0%** | 11.0% | 74.0% | 46.48 | 4648 | 1904 |

### Samples Per Problem Sweep (temp=0.5, max_steps=10)

| Samples | Total Traces | Accuracy | Step Validity | Valid Traces | Diversity | Avg Steps |
|---------|--------------|----------|---------------|--------------|-----------|-----------|
| 2 | 100 | 95.0% | 36.2% | 11.0% | 78.0% | 48.75 |
| **4** | 200 | **96.0%** | 35.3% | 9.0% | **94.0%** | 47.01 |

---

## JSON Data (for plotting)

```json
[
  {
    "config": "temp=0.0",
    "temperature": 0.0,
    "max_steps": 10,
    "samples_per_problem": 2,
    "total_traces": 100,
    "correct": 95,
    "accuracy": 95.0,
    "valid_traces": 2,
    "valid_trace_pct": 2.0,
    "total_steps": 5161,
    "valid_steps": 1405,
    "step_validity": 27.22,
    "avg_steps": 51.61,
    "diverse_problems": 3,
    "same_problems": 47,
    "diversity_pct": 6.0
  },
  {
    "config": "temp=0.3",
    "temperature": 0.3,
    "max_steps": 10,
    "samples_per_problem": 2,
    "total_traces": 100,
    "correct": 96,
    "accuracy": 96.0,
    "valid_traces": 6,
    "valid_trace_pct": 6.0,
    "total_steps": 4857,
    "valid_steps": 1322,
    "step_validity": 27.22,
    "avg_steps": 48.57,
    "diverse_problems": 29,
    "same_problems": 21,
    "diversity_pct": 58.0
  },
  {
    "config": "temp=0.5",
    "temperature": 0.5,
    "max_steps": 10,
    "samples_per_problem": 2,
    "total_traces": 100,
    "correct": 95,
    "accuracy": 95.0,
    "valid_traces": 11,
    "valid_trace_pct": 11.0,
    "total_steps": 4875,
    "valid_steps": 1764,
    "step_validity": 36.18,
    "avg_steps": 48.75,
    "diverse_problems": 39,
    "same_problems": 11,
    "diversity_pct": 78.0
  },
  {
    "config": "temp=0.7",
    "temperature": 0.7,
    "max_steps": 10,
    "samples_per_problem": 2,
    "total_traces": 100,
    "correct": 87,
    "accuracy": 87.0,
    "valid_traces": 6,
    "valid_trace_pct": 6.0,
    "total_steps": 4574,
    "valid_steps": 1617,
    "step_validity": 35.35,
    "avg_steps": 45.74,
    "diverse_problems": 41,
    "same_problems": 9,
    "diversity_pct": 82.0
  },
  {
    "config": "temp=1.0",
    "temperature": 1.0,
    "max_steps": 10,
    "samples_per_problem": 2,
    "total_traces": 100,
    "correct": 80,
    "accuracy": 80.0,
    "valid_traces": 10,
    "valid_trace_pct": 10.0,
    "total_steps": 3763,
    "valid_steps": 1884,
    "step_validity": 50.07,
    "avg_steps": 37.63,
    "diverse_problems": 43,
    "same_problems": 7,
    "diversity_pct": 86.0
  },
  {
    "config": "max_steps=5",
    "temperature": 0.5,
    "max_steps": 5,
    "samples_per_problem": 2,
    "total_traces": 100,
    "correct": 90,
    "accuracy": 90.0,
    "valid_traces": 13,
    "valid_trace_pct": 13.0,
    "total_steps": 2654,
    "valid_steps": 909,
    "step_validity": 34.25,
    "avg_steps": 26.54,
    "diverse_problems": 37,
    "same_problems": 13,
    "diversity_pct": 74.0
  },
  {
    "config": "max_steps=15",
    "temperature": 0.5,
    "max_steps": 15,
    "samples_per_problem": 2,
    "total_traces": 100,
    "correct": 93,
    "accuracy": 93.0,
    "valid_traces": 11,
    "valid_trace_pct": 11.0,
    "total_steps": 4648,
    "valid_steps": 1904,
    "step_validity": 40.96,
    "avg_steps": 46.48,
    "diverse_problems": 37,
    "same_problems": 13,
    "diversity_pct": 74.0
  },
  {
    "config": "samples=4",
    "temperature": 0.5,
    "max_steps": 10,
    "samples_per_problem": 4,
    "total_traces": 200,
    "correct": 192,
    "accuracy": 96.0,
    "valid_traces": 18,
    "valid_trace_pct": 9.0,
    "total_steps": 9402,
    "valid_steps": 3318,
    "step_validity": 35.29,
    "avg_steps": 47.01,
    "diverse_problems": 47,
    "same_problems": 3,
    "diversity_pct": 94.0
  }
]
```

---

## Metric Definitions

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | `correct / total_traces × 100` | Final answer matches ground truth label |
| **Step Validity** | `valid_steps / total_steps × 100` | Individual reasoning steps verified by solver |
| **Valid Traces** | `fully_valid / total_traces × 100` | Traces where ALL steps are solver-valid |
| **Diversity** | `diverse_problems / total_problems × 100` | Problems where 2 samples differ (answer or steps) |
| **Avg Steps** | `sum(steps) / total_traces` | Average reasoning steps per trace |

---

## Key Observations

### 1. Temperature vs Accuracy/Diversity Trade-off

```
Accuracy:   95% → 96% → 95% → 87% → 80%  (decreases with temp)
Diversity:   6% → 58% → 78% → 82% → 86%  (increases with temp)
```

**Sweet spot**: Temperature 0.5 achieves 95% accuracy with 78% diversity.

### 2. Temperature vs Step Validity (Surprising!)

```
Step Validity: 27% → 27% → 36% → 35% → 50%
```

Higher temperature actually improves step validity! This may be because:
- Greedy decoding gets "stuck" in repetitive patterns
- Sampling explores different valid reasoning paths
- Shorter traces at high temp (37 vs 52 steps) means fewer chances for errors

### 3. Max Steps Effect

```
Steps=5:  90% accuracy, 34% validity, 27 avg steps
Steps=10: 95% accuracy, 36% validity, 49 avg steps  
Steps=15: 93% accuracy, 41% validity, 46 avg steps
```

Longer max_steps allows model to reach correct conclusions but may generate redundant steps.

### 4. DPO Requirements

For DPO training, we need:
- **High diversity** (≥70%) - different samples for preference pairs
- **Mixed validity** - some valid, some invalid for learning
- **Reasonable accuracy** (≥85%) - mostly correct conclusions

**Best config**: temp=0.5, max_steps=15, samples=2

---

## Recommended Configuration

```yaml
# configs/trace_generation.yaml
temperature: 0.5        # Balance accuracy (95%) and diversity (78%)
max_steps: 15           # Higher step validity (41%)
samples_per_problem: 2  # Sufficient for DPO pairs
top_p: 0.95            # Standard nucleus sampling
max_tokens: 2048       # Sufficient for multi-step reasoning
```

---

## Plotting Scripts

### Python (matplotlib)

```python
import matplotlib.pyplot as plt
import json

# Load data
data = json.loads(open("docs/18_hyperparameter_search.md").read().split("```json")[1].split("```")[0])

# Temperature sweep
temp_data = [d for d in data if d["config"].startswith("temp=")]
temps = [d["temperature"] for d in temp_data]
acc = [d["accuracy"] for d in temp_data]
div = [d["diversity_pct"] for d in temp_data]
step_val = [d["step_validity"] for d in temp_data]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(temps, acc, 'o-', label="Accuracy (%)", color="blue")
ax.plot(temps, div, 's-', label="Diversity (%)", color="green")
ax.plot(temps, step_val, '^-', label="Step Validity (%)", color="orange")
ax.set_xlabel("Temperature")
ax.set_ylabel("Percentage")
ax.set_title("Temperature Effect on Trace Generation Quality")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("paper/figures/temp_sweep.pdf")
```

### CSV Export

```csv
config,temperature,max_steps,samples,accuracy,step_validity,valid_traces,diversity,avg_steps
temp=0.0,0.0,10,2,95.0,27.22,2.0,6.0,51.61
temp=0.3,0.3,10,2,96.0,27.22,6.0,58.0,48.57
temp=0.5,0.5,10,2,95.0,36.18,11.0,78.0,48.75
temp=0.7,0.7,10,2,87.0,35.35,6.0,82.0,45.74
temp=1.0,1.0,10,2,80.0,50.07,10.0,86.0,37.63
max_steps=5,0.5,5,2,90.0,34.25,13.0,74.0,26.54
max_steps=15,0.5,15,2,93.0,40.96,11.0,74.0,46.48
samples=4,0.5,10,4,96.0,35.29,9.0,94.0,47.01
```

