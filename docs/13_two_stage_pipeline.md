# SOKRATES Two-Stage Pipeline

The OaK-DPO training is split into two stages for **better optimization and clearer logging** - the iterative "fresh traces each round" algorithm stays the same.

## Key Principle: Fresh Traces Each Iteration

```
Iteration 1: Model_SFT  → Traces₁ → DPO → Model₁
Iteration 2: Model₁     → Traces₂ → DPO → Model₂  (NEW traces with improved model!)
Iteration 3: Model₂     → Traces₃ → DPO → Model₃  (NEW traces with even better model!)
```

Each iteration generates **fresh traces** because the improved model produces better (more valid) traces, leading to better preference pairs.

---

## Stage 1: Trace Generation

Generate and verify optionized reasoning traces.

### Script: `scripts/generate_traces.py`

```bash
# Single GPU - Quick test
CUDA_VISIBLE_DEVICES=0 python scripts/generate_traces.py \
    --model outputs/sft/20251209_150417/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/test \
    --num-problems 100 \
    --samples-per-problem 2

# Full run - 1500 problems
CUDA_VISIBLE_DEVICES=0 python scripts/generate_traces.py \
    --model outputs/sft/20251209_150417/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/full \
    --num-problems 1500 \
    --samples-per-problem 2 \
    --max-steps 6

# Multi-GPU (experimental)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/generate_traces.py \
    --model outputs/sft/20251209_150417/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/full \
    --num-problems 1500 \
    --samples-per-problem 2 \
    --num-gpus 4
```

### Output Files

```
outputs/traces/run1/
├── traces.jsonl          # All generated traces
├── summary.json          # Generation statistics
└── generation_config.json # Config used
```

### Trace Format (traces.jsonl)

```json
{
  "problem_id": "prontoqa_train_001",
  "label": "TRUE",
  "final_answer": "TRUE",
  "correct": true,
  "num_steps": 4,
  "steps": [
    {
      "step_idx": 0,
      "thought": "The premise states...",
      "action": "<Option type=\"SUBST\" .../>",
      "option_type": "SUBST",
      "option_args": [0, "Alex", "mammal"],
      "solver_valid": true,
      "solver_error": null
    }
  ],
  "all_steps_valid": true,
  "valid_step_count": 4,
  "total_step_count": 4
}
```

---

## Stage 2: DPO Training

Train DPO on preference pairs built from traces.

### Script: `scripts/train_dpo_from_traces.py`

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train_dpo_from_traces.py \
    --traces outputs/traces/full/traces.jsonl \
    --model outputs/sft/20251209_150417/final \
    --output outputs/dpo/run1 \
    --num-epochs 1 \
    --batch-size 2 \
    --beta 0.1

# Multi-GPU with accelerate
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/full/traces.jsonl \
    --model outputs/sft/20251209_150417/final \
    --output outputs/dpo/run1 \
    --num-epochs 1 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --wandb
```

### Output Files

```
outputs/dpo/run1/
├── final/                  # Final DPO model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── preference_pairs.jsonl  # Pairs used for training
├── dpo_config.json         # Training config
├── training_summary.json   # Results
└── checkpoint-*/           # Intermediate checkpoints
```

---

## Quick Start: Full OaK Loop

```bash
# Run 2 iterations with 1500 problems on GPU 0
./scripts/run_oak_loop.sh 2 1500 "0"

# Run 2 iterations on multiple GPUs
./scripts/run_oak_loop.sh 2 1500 "0,1,2,3"
```

This script handles the full iterative loop automatically, using fresh traces each iteration.

---

## Manual Workflow (Step by Step)

```bash
# 1. Generate traces (inference only)
CUDA_VISIBLE_DEVICES=0 python scripts/generate_traces.py \
    --model outputs/sft/20251209_150417/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/iter0 \
    --num-problems 1500 \
    --samples-per-problem 2

# 2. Inspect traces (optional)
head -5 outputs/traces/iter0/traces.jsonl | python -m json.tool
cat outputs/traces/iter0/summary.json

# 3. Train DPO
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter0/traces.jsonl \
    --model outputs/sft/20251209_150417/final \
    --output outputs/dpo/iter0 \
    --wandb

# 4. Evaluate
python scripts/evaluate.py \
    --model outputs/dpo/iter0/final \
    --data data/processed/prontoqa_test.jsonl \
    --merge-adapter

# 5. (Optional) Iterate - generate new traces with DPO model
CUDA_VISIBLE_DEVICES=0 python scripts/generate_traces.py \
    --model outputs/dpo/iter0/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/iter1 \
    --num-problems 1500 \
    --samples-per-problem 2

# 6. Train DPO again on new traces
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 --mixed_precision=bf16 \
    scripts/train_dpo_from_traces.py \
    --traces outputs/traces/iter1/traces.jsonl \
    --model outputs/dpo/iter0/final \
    --output outputs/dpo/iter1 \
    --wandb
```

---

## Benefits of Two-Stage Approach

| Aspect | Coupled (old) | Decoupled (new) |
|--------|---------------|-----------------|
| **Debugging** | Hard - everything in one loop | Easy - inspect traces before DPO |
| **Optimization** | One-size-fits-all | Tailor each stage separately |
| **Reusability** | Re-run everything | Re-train DPO on same traces |
| **Checkpointing** | Only at iteration end | Save traces independently |
| **GPU utilization** | Mixed inference/training | Dedicated per stage |

---

## Recommended Settings

### Trace Generation (Inference)

| Setting | Toy Test | Full Run |
|---------|----------|----------|
| num-problems | 50 | 1500 |
| samples-per-problem | 2 | 2-4 |
| max-steps | 4 | 6 |
| max-thought-tokens | 40 | 60 |
| temperature | 0.0 (greedy) | 0.0 or 0.7 |

### DPO Training

| Setting | Toy Test | Full Run |
|---------|----------|----------|
| num-epochs | 1 | 1-2 |
| batch-size | 2 | 4-8 |
| learning-rate | 5e-6 | 5e-6 |
| beta | 0.1 | 0.1 |
| gradient-accumulation-steps | 2 | 4-8 |

---

## Troubleshooting

**No preference pairs created:**
- Check trace validity in summary.json
- Lower `--min-validity-gap` to 0.0
- Use `--require-valid-winner=false`

**OOM during trace generation:**
- Reduce `--samples-per-problem`
- Use single GPU with `device_map="auto"`

**OOM during DPO:**
- Reduce `--batch-size` to 1
- Increase `--gradient-accumulation-steps`
- Enable gradient checkpointing in model

**Slow trace generation:**
- Use greedy decoding (`--temperature 0.0`)
- Reduce `--max-steps`
- Consider vLLM for faster inference

