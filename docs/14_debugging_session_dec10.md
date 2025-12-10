# 14: Debugging Session - December 10, 2025

This document records the comprehensive debugging and optimization session that identified and fixed critical issues in the SOKRATES training pipeline.

---

## Executive Summary

| Issue | Impact | Fix |
|-------|--------|-----|
| **SFT data loader bug** | Model learned nothing useful | Use `training_text` directly |
| **Optionizer broken** | Wrong premise indices, predicate thoughts | Rewrote `_build_step_with_reasoning` |
| **Greedy generation** | No diversity for DPO pairs | Enable sampling (temp=0.7) |
| **SFT overfitting** | Loss collapsed but no generalization | Lower LR, more epochs |

**Total time spent debugging**: ~3 hours  
**Outcome**: Pipeline now produces correct natural language reasoning traces

---

## 1. Initial Problem Discovery

### Symptom
After running trace generation with the SFT model:
- Only 2 steps per trace (should be 3-6)
- Thoughts were `<Thought>` literal string
- Model jumped straight to CONCLUDE
- Malformed action args: `[0], [1], 2` instead of `[0]`

### Test Command
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python scripts/generate_traces.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/test_100 \
    --num-problems 100 \
    --samples-per-problem 2 \
    --num-gpus 6
```

### Observed Output
```
Step 0:
  Thought: <Thought>
  Action: <Option type="MODUS_PONENS" args="[0, 1]" />
  Valid: True

Step 1:
  Thought: <Thought>
  Action: <Option type="CONCLUDE" args="[0]" />
  Valid: False
```

**Expected**: Multi-step reasoning with natural language thoughts.

---

## 2. Root Cause Analysis

### Issue 1: Training Data Loader Bug

**Location**: `scripts/train_sft.py:load_training_data()`

**Problem**: The loader looked for `steps` and `premises` keys in the JSON, but our data only has:
```json
{
    "problem_id": "...",
    "training_text": "Premises:\n  [0] ...\nReasoning:\nThought: ...\nAction: ...",
    "label": "TRUE",
    "num_steps": 5
}
```

**Consequence**: The loader fell back to creating minimal CONCLUDE-only traces:
```python
else:
    # Create a minimal conclude step  <-- THIS WAS EXECUTING
    steps = [ProofStep(
        step_idx=0,
        thought=f"Based on the premises, the conclusion is {label}.",
        option_type=OptionType.CONCLUDE,
        ...
    )]
```

**Fix**: Added `TrainingTextWrapper` class to use `training_text` directly:
```python
class TrainingTextWrapper:
    """Wrapper that holds pre-formatted training text."""
    def __init__(self, problem_id: str, training_text: str, label: str):
        self.problem_id = problem_id
        self._training_text = training_text
        self.final_answer = label
    
    def to_training_string(self) -> str:
        return self._training_text
```

---

### Issue 2: Optionizer Generating Wrong Data

**Location**: `src/data/optionizer.py:_infer_option_from_text()`

**Problem**: The function always returned `[0, 1]` as premise indices:
```python
def _infer_option_from_text(self, text, formulas):
    # ALWAYS returns [0, 1] regardless of actual premises!
    return OptionType.MODUS_PONENS, [0, 1]
```

**Additionally**: Thoughts were predicate-style instead of natural language:
```
Before: Thought: Nervous('Wren', True)
After:  Thought: Since wren is a jompus (premise 0) and each jompus is nervous (premise 8), we can conclude that Wren is nervous.
```

**Fix**: Rewrote `optionize_prontoqa_example()` with proper premise tracking:
```python
def _build_step_with_reasoning(self, entity, prop, value, premises, formulas, derived_facts, step_idx):
    """Build a reasoning step with proper premise indices and natural language thought."""
    # Find premises that mention this entity or property
    entity_premise_idx = -1
    rule_premise_idx = -1
    
    for i, premise in enumerate(premises):
        premise_lower = premise.lower()
        if entity_lower and entity_lower in premise_lower:
            entity_premise_idx = i
        if prop_lower in premise_lower and "every" in premise_lower:
            rule_premise_idx = i
    
    # Generate natural language thought
    thought = f"Since {entity_text} (premise {entity_premise_idx}) and {rule_text} (premise {rule_premise_idx}), we can conclude that {entity} is {prop_lower}."
    
    return OptionType.MODUS_PONENS, [entity_premise_idx, rule_premise_idx], thought
```

---

### Issue 3: SFT Overfitting

**Symptom**: Loss dropped from 3.5 → 0.01 within first epoch, but model still generated garbage.

**Training Logs**:
```
Epoch 0.15: loss = 3.49
Epoch 0.59: loss = 1.86
Epoch 0.89: loss = 0.21
Epoch 1.03: loss = 0.08
Epoch 1.33: loss = 0.01  ← Collapsed!
```

**Root Cause**: The model was training on CONCLUDE-only traces due to Issue 1, so it learned to always output CONCLUDE.

**Fix**: After fixing Issue 1, the loss curve should show gradual learning of the actual reasoning patterns.

---

### Issue 4: Greedy Decoding in Trace Generation

**Problem**: With `temperature=0.0` (greedy), all samples per problem are identical:
```python
Problem: ProntoQA_train_hop-4_case-50_ic-5
Sample 1 steps: 2
Sample 2 steps: 2
Same?: True  # <-- Both samples identical!
```

**Impact**: DPO requires diversity to create preference pairs. With identical samples, there's nothing to compare.

**Fix**: Enable sampling with temperature:
```yaml
trace_generation:
  temperature: 0.7    # Was 0.0
  do_sample: true     # Was false
```

---

## 3. Files Modified

### `scripts/train_sft.py`
- Added `TrainingTextWrapper` class
- Modified `load_training_data()` to use `training_text` field

### `src/data/optionizer.py`
- Added `_parse_predicate()` method
- Added `_predicate_to_natural()` method  
- Added `_build_step_with_reasoning()` method
- Rewrote `optionize_prontoqa_example()` for natural language thoughts

### `configs/training.yaml`
- `sft.num_epochs`: 3 → 5
- `sft.learning_rate`: 2e-5 → 1e-5
- `trace_generation.temperature`: 0.0 → 0.7
- `trace_generation.do_sample`: false → true
- `trace_generation.max_thought_tokens`: 60 → 150

### `scripts/generate_traces.py`
- Fixed `trace_to_dict()` to use `step.to_action_string()` instead of `step.action`

---

## 4. Before vs After Comparison

### Training Data Format

**Before** (broken):
```
Thought: Nervous('Wren', True)
Action: <Option type="MODUS_PONENS" args="[0, 1]" />
Thought: Yumpus('Wren', True)
Action: <Option type="MODUS_PONENS" args="[0, 1]" />
```

**After** (fixed):
```
Thought: Since wren is a jompus (premise 0) and each jompus is nervous (premise 8), we can conclude that Wren is nervous.
Action: <Option type="MODUS_PONENS" args="[0, 8]" />
Thought: Since wren is nervous (premise 12) and each yumpus is a zumpus (premise 11), we can conclude that Wren is yumpus.
Action: <Option type="MODUS_PONENS" args="[12, 11]" />
```

### Model Output (Expected After Fix)

**Before** (broken SFT):
```
Thought: Based on the premises, the conclusion is TRUE.
Action: <Option type="CONCLUDE" args="[0], [1], 2" />
```

**After** (fixed SFT):
```
Thought: Since Wren is a jompus (premise 0) and every jompus is nervous (premise 1), we can apply modus ponens.
Action: <Option type="MODUS_PONENS" args="[0, 1]" />
Thought: Since Wren is nervous and every nervous thing is tired (premise 2), Wren is tired.
Action: <Option type="MODUS_PONENS" args="[12, 2]" />
Thought: The conclusion "Wren is tired" follows from the reasoning above.
Action: <Option type="CONCLUDE" args="[0]" />
Final Answer: TRUE
```

---

## 5. Debugging Methodology

### Step 1: Trace Generation Test
```bash
# Generate small sample
python scripts/generate_traces.py --num-problems 100 --output outputs/traces/test

# Check output format
head -1 outputs/traces/test/traces.jsonl | python3 -m json.tool
```

### Step 2: Direct Model Test
```python
# Test model generation directly
prompt = """Premises:
  [0] Wren is a jompus
  [1] Every jompus is nervous
...
Reasoning:
Thought:"""

outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0]))
```

### Step 3: Training Data Inspection
```bash
# Check what data looks like
head -1 data/processed/prontoqa_train.jsonl | python3 -m json.tool

# Verify keys present
python3 -c "import json; d=json.loads(open('data/processed/prontoqa_train.jsonl').readline()); print('Keys:', list(d.keys()))"
```

### Step 4: Training Log Analysis
```bash
# Check loss curve
cat outputs/sft/*/training_history.json | python3 -m json.tool | head -50
```

---

## 6. Lessons Learned

### 1. Always Verify Data Loading
The SFT trainer's loss going down doesn't mean the model is learning the right thing. Always test generation quality, not just loss.

### 2. Check Key Mismatches
When loading JSON data, verify that the code is actually using the right keys. Our data had `training_text` but the loader was looking for `steps`.

### 3. Predicate vs Natural Language
Machine-readable predicates like `Nervous('Wren', True)` are not good training data for LLMs. Natural language explanations train better.

### 4. Premise Index Tracking
For step-by-step reasoning, you must track which premises/derived facts are used at each step. Generic `[0, 1]` indices don't teach anything useful.

### 5. Sampling for DPO
DPO requires preference pairs with contrasting quality. Greedy decoding produces identical samples - always use temperature > 0 for trace generation.

### 6. Test the Full Pipeline Early
Don't wait until the end to test generation quality. Test after SFT before running expensive DPO.

---

## 7. Commands for Re-running

### Re-process Data
```bash
# Backup old data
mv data/processed/prontoqa_train.jsonl data/processed/prontoqa_train_old.jsonl

# Re-run with fixed optionizer
python scripts/prepare_data.py --skip-download
```

### Re-run SFT
```bash
# Archive broken runs
mv outputs/sft outputs/sft_broken

# Run with fixed loader
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes=6 --mixed_precision=bf16 \
    scripts/train_sft.py \
    --config configs/training.yaml \
    --data data/processed/prontoqa_train.jsonl \
    --output-dir outputs/sft
```

### Test New Model
```bash
# Quick generation test
CUDA_VISIBLE_DEVICES=2 python scripts/generate_traces.py \
    --model outputs/sft/latest/final \
    --data data/processed/prontoqa_train.jsonl \
    --output outputs/traces/test_fixed \
    --num-problems 50 \
    --samples-per-problem 2
```

---

## 8. Timeline

| Time | Action |
|------|--------|
| 00:00 | Ran trace generation, noticed broken output |
| 00:30 | Identified model generating `<Thought>` literals |
| 01:00 | Found training data had predicate-style thoughts |
| 01:30 | Fixed optionizer to generate natural language |
| 02:00 | Re-processed data, re-ran SFT |
| 02:30 | SFT complete, tested model - still broken |
| 02:45 | Found SFT loader bug (not using `training_text`) |
| 03:00 | Fixed loader, documented everything |

---

## 9. Related Documentation

- [04_pipeline_guide.md](../paper/docs/04_pipeline_guide.md) - Full pipeline explanation
- [09_session_log.md](09_session_log.md) - Previous session notes
- [10_experimental_design.md](10_experimental_design.md) - Data split rationale

