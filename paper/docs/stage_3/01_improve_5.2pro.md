## High-impact fixes

### 1) Correct (or rephrase) the “33× improvement” claim

Your abstract says “full-trace validity from 2.1% to 92.0%—a 33× improvement” . But 92.0 / 2.1 ≈ **43.8×**, not 33×. If you want to keep “33×”, it matches the *iter-1* jump 2.1% → 71.3% (≈34×) described later .
Fix options:

* **Option A (precise):** “2.1% → 92.0% (≈44×)”
* **Option B (safer):** replace the multiplier with **absolute points**: “2.1% → 92.0% (+89.9 pts)”.

### 2) Don’t count “unparsable steps” as valid (or at least report sensitivity)

The verifier section currently says: “If a step cannot be parsed for verification, we accept it as valid” . This is a major threat to validity because it can inflate step/trace validity, especially if the model learns to exploit parser blind spots.

Minimum improvements:

* Report **parse coverage**: % of steps parsed successfully (PrOntoQA) and % of steps either parsed or supported (FOLIO).
* Report **two metrics**:

  * *Optimistic*: unparsable → valid (current).
  * *Conservative*: unparsable → invalid (or “unknown”, reported separately).
* Add a short subsection: “Verifier coverage and metric sensitivity”.

If you keep the current choice, it needs to be **front-and-center** (abstract or limitations), not buried.

### 3) Fix broken cross-reference / placeholder text

You have: “Our experiments confirm that the Sokrates loop produces such calibration (Section ).” 
That reads like a missing section number. This is an easy credibility hit—fix before submission.

### 4) Tighten the “first closed-loop OaK instantiation” claim

You say “To our knowledge, Sokrates represents the first closed-loop instantiation of the OaK framework in a neural language model.” 
Even if true, reviewers often push back unless you define “closed-loop OaK instantiation” sharply.

Safer wording:

* “To our knowledge, this is the first **LLM reasoning** system that (i) represents proofs as a fixed **option vocabulary**, (ii) learns an explicit **option-success predictor**, and (iii) aligns the option policy using **solver-derived DPO preferences** in an iterative loop.”

That anchors the novelty in concrete design choices you already describe .

---

## Method clarity upgrades that will reduce reviewer confusion

### 5) Make the option/state formalization “one-screen understandable”

You already have the OaK mapping and the 11-rule option vocabulary . What’s missing is a short “Putting it together” paragraph immediately after Table 2 that answers:

* What exactly is fed into the LM at step *t* (full prompt with numbered premises + derived list + goal)?
* What exactly is emitted (Thought tokens + constrained Action tokens)?
* What are the *typed* arguments and how they map into the formula list?

You do describe the formula list indexing later —pull that explanation earlier and make it explicit.

### 6) Separate “syntax validity” from “semantic validity” explicitly

You mention constrained decoding and that grammar eliminates syntax errors , but you should elevate it into a clean two-line definition:

* **Syntax-valid** action: conforms to Option grammar (guaranteed by decoding).
* **Solver-valid** action: accepted by solver/verifier given current state.

This will help readers interpret step validity numbers.

### 7) Preference construction: justify the scoring weights (or remove them)

Your preference score mixes step-validity rate + correctness + full-trace bonus . Reviewers will ask why “+0.5” and why this exact structure.

Quick improvements:

* Add a short ablation: weights {0.25, 0.5, 1.0} for the trace-validity bonus.
* Or simplify: lexicographic ranking (1) fully valid & correct, then (2) correct, then (3) step-validity rate.

---

## Experimental upgrades that strengthen the paper materially

### 8) Report depth-stratified results (you already hint at it)

You note trace validity drops with proof depth (1–2 steps: 99.8% vs 4–5 steps: 82.3%) . Turn that into a figure/table:

* Accuracy / Step validity / Trace validity by depth (1…5)
* Include SFT vs iter1 vs iter2

This is a high-value analysis that costs little.

### 9) Add “verifier coverage” stats for FOLIO transfer

On FOLIO you mark unsupported rules as UNKNOWN and count them invalid , which can depress step/trace validity. Add:

* % of steps using supported option types
* % of steps requiring Z3 entailment vs rule checks
* timeout rate (you mention a 5s timeout) 

Without this, the transfer validity numbers are hard to interpret.

### 10) Add one more baseline: “DPO without q̂ϕ head” (training-only head ablation)

You emphasize the option-success predictor q̂ϕ as explicit knowledge . Right now, it’s not isolated as a causal contributor.

Add ablations:

* DPO with solver preferences **without** training q̂ϕ
* q̂ϕ trained **without** DPO (frozen policy)

You already have a key ablation “w/o solver (answer-only)” showing solver supervision matters . The next natural reviewer question is whether q̂ϕ adds anything beyond DPO.

### 11) Fix reproducibility expectations

You list model/training hyperparameters and hardware  and timing , which is good, but add:

* random seeds
* exact evaluation decoding parameters (you partially do) 
* whether solver/verifier code will be released

---

## Writing and positioning improvements

### 12) Make the abstract less “gotcha”, more precise about what is measured

The abstract hook is strong , but it implicitly suggests “94% accuracy” is typical of generic CoT, when in your results it’s the *optionized SFT baseline* that hits 94.2% .

Clarify early:

* “In our optionized SFT baseline…” or
* “Even when answer accuracy is high…”

### 13) Rename “baby OaK cycle”

You use “baby OaK cycle” ; it reads informal. Replace with “micro-OaK loop” consistently (you already use that elsewhere).

---

## Drop-in rewritten abstract (copy/paste)

> **Abstract (revised)**
> Large language models can achieve high answer accuracy on logical reasoning benchmarks while producing chains of thought containing invalid inference steps. In our optionized supervised fine-tuning (SFT) baseline on PrOntoQA, answer accuracy reaches 94.2% yet only 2.1% of traces are fully solver-valid.  We present **Sokrates**, an Options-and-Knowledge (OaK) instantiation for first-order logic reasoning that represents proofs as sequences of discrete inference-rule **options** verified step-by-step by a symbolic solver.  From solver feedback, Sokrates (i) trains a calibrated **option-success predictor** ( \hat{q}_\phi(s,\omega) ) and (ii) constructs solver-derived preference pairs for **Direct Preference Optimization**, aligning the option policy toward stepwise validity.  On PrOntoQA, Sokrates improves accuracy to 97.6% and raises step validity from 27.3% to 98.5% and trace validity from 2.1% to 92.0%.  The learned predictor is well calibrated (ECE 0.08), and policies transfer zero-shot to FOLIO, improving accuracy from 45.3% to 53.2%. 

Notes:

* This version removes the incorrect “33×” multiplier and makes the “94% accuracy” attribution precise.

---

## Drop-in “Threats to validity” paragraph (recommended)

You currently acknowledge parser limitations but in a way that can be read as metric leakage . Consider adding:

> **Threats to validity.** Our step-level verification depends on dataset-specific parsing and rule checking. For PrOntoQA, some steps may be unparsable by the current verifier; we report verifier coverage and evaluate both optimistic and conservative scoring variants (treating unparsable steps as valid vs invalid) to quantify sensitivity.  For FOLIO, our rule-level verifier supports only a subset of option types and uses Z3 entailment checks with timeouts; we therefore report the fraction of steps covered by each verification mode in addition to step/trace validity.

---

If you want the fastest path to a stronger submission: fix the unparsable-step accounting, correct the “33×” statement, and add verifier coverage/sensitivity + depth-stratified results. Those changes directly address the biggest reasons reviewers would discount the impressive trace-validity numbers.
