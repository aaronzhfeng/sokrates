I’m going to take the upgrades from the feedback (DPO-centric, constrained decoding, PrOntoQA curriculum, explicit option‑knowledge head, hybrid Thought/Action format). They fix real failure modes and don’t blow up the engineering cost, so there’s no good reason not to use them.

Below is a **concise, paper-ready plan**.

---

## Title

**OaK‑DPO: Distilling Symbolic Knowledge into Option‑Level Reasoning via Solver‑Guided Preference Optimization**

---

## Abstract (≈180–200 words)

Large language models (LLMs) still produce logically invalid chain‑of‑thought (CoT) reasoning, even when final answers are correct. Recent neuro‑symbolic work such as LoCo‑LMs and VeriCoT uses logical constraints or first‑order logic (FOL) solvers to improve consistency, but typically treats reasoning as unstructured text and focuses solely on better CoT traces rather than explicit knowledge about reasoning actions.

We propose **OaK‑DPO**, a small but concrete instantiation of Sutton’s *Options and Knowledge* (OaK) architecture in a logical micro‑world.  We represent proofs on FOLIO and P‑FOLIO as sequences of discrete reasoning **options**—inference rule macros such as `MODUS_PONENS` or `UNIV_INSTANTIATION` with arguments—using human‑annotated proof chains from P‑FOLIO.  A first‑order solver, available via FOL annotations and the PrOntoQA synthetic environment, provides ground‑truth **knowledge** by verifying whether each option application is logically valid.

From solver labels we build (i) an explicit option‑success predictor (\hat{q}*\phi(s,\omega)) and (ii) a preference dataset over full optionized traces, where solver‑valid proofs are preferred over invalid ones. We then apply Direct Preference Optimization (DPO) to align a 7–8B LLM’s option policy with these solver‑induced preferences.  Experiments on FOLIO and PrOntoQA show that OaK‑DPO improves final accuracy, full‑trace logical validity, and calibration of (\hat{q}*\phi) compared to supervised fine‑tuning, a semantic‑loss baseline, and a VeriCoT‑style solver‑validated CoT approach. We argue this constitutes an OaK‑style loop for logical reasoning, where options and knowledge are both learned from experience.

---

## Keywords

* neuro‑symbolic language models
* logical reasoning
* chain‑of‑thought (CoT)
* Direct Preference Optimization (DPO)
* Options and Knowledge (OaK)
* FOLIO / P‑FOLIO
* PrOntoQA
* constrained decoding

---

## Paper structure

### 1. Introduction

* Motivation: LLMs are logically inconsistent on FOL benchmarks like FOLIO; simple CoT is not enough.
* Limitations of prior neuro‑symbolic work: LoCo‑LM (semantic loss) and VeriCoT (step‑wise verification) do not model options or predictive knowledge.
* Our proposal: OaK‑DPO = explicit reasoning options + solver‑derived knowledge + DPO alignment.

### 2. Background

* OaK: options as temporally extended behaviours; knowledge as predictive models / GVFs; reward‑respecting subtasks.
* FOLIO, P‑FOLIO, PrOntoQA: FOLIO’s NL+FOL annotations, P‑FOLIO’s step‑wise proofs, PrOntoQA’s synthetic QA world.
* LoCo‑LMs and VeriCoT as closest baselines.
* DPO for preference alignment.

### 3. Problem setup and OaK formulation

* Logical reasoning tasks: premises + conclusion → label.
* **Options**: define an option vocabulary from P‑FOLIO’s rule tags; each option is a cognitive macro: Thought (NL justification) + Action `<Option type=… args=…>`.
* **Knowledge**: solver defines whether applying an option in state (s) leads to a valid next state; option‑success probabilities (\hat{q}_\phi(s,\omega)) approximate this.
* Micro OaK loop: generate experience → solver labels → update (\hat{q}_\phi) and policy.

### 4. Method: OaK‑DPO

4.1 **Optionized CoT format**

* Describe hybrid output: “Thought: … Action: <Option …>”.
* Use constrained decoding (e.g., Outlines / similar) so option type and arguments are always syntactically valid (indices from current formula set).

4.2 **Solver‑guided verification**

* Use FOLIO’s FOL annotations and a FOL solver; for PrOntoQA, use its synthetic ontology engine.
* For each step: check entailment; for each trace: mark VALID if all steps valid and label correct.

4.3 **Option models (knowledge)**

* Define (\hat{q}_\phi(s,\omega)) to predict step validity; train with cross‑entropy on solver labels.
* Optionally add 1–2 simple “reward‑respecting subtasks” (e.g., consistency, goal proximity) as GVF‑style heads.

4.4 **Preference construction and DPO**

* For each task, sample multiple optionized traces from current policy.
* Build (winner, loser) pairs where winners are solver‑valid proofs and losers contain invalid steps.
* Apply DPO (using TRL/HF) to a 7–8B model with LoRA, aligning the option policy to solver‑induced preferences.

4.5 **Micro OaK iterations**

* Run 2–3 outer cycles: SFT → generate+verify → DPO → repeat, to demonstrate a small OaK‑style loop.

### 5. Experimental setup

* **Datasets:**

  * Stage 1: PrOntoQA for large‑scale synthetic pretraining of options and semantics.
  * Stage 2: FOLIO/P‑FOLIO for adaptation to NL proofs.

* **Models & baselines:**

  * Base 7–8B instruction model.
  * Baselines: base CoT, SFT on P‑FOLIO, LoCo‑style semantic loss, VeriCoT‑style solver‑validated CoT.

* **Metrics:** final answer accuracy; step validity rate; full‑trace validity; calibration of (\hat{q}_\phi) (Brier / ECE).

### 6. Results

* Show OaK‑DPO improves:

  * full‑trace validity and final accuracy vs SFT and baselines;
  * calibration curves of (\hat{q}_\phi) across training epochs (knowledge actually improves).

### 7. Analysis and ablations

* Impact of constrained vs unconstrained decoding (syntax vs logic errors).
* DPO on raw CoT vs optionized CoT.
* With vs without explicit option‑knowledge head.
* Transfer from PrOntoQA to FOLIO.

### 8. Related work and conclusion

* Position relative to LoCo‑LMs, VeriCoT, other solver‑augmented CoT and neurosymbolic RL.
* Summarize contributions and outline extensions toward fuller OaK (richer option models, continual learning, temporal reasoning).

---

## Implementation plan (compressed, realistic)

**Phase 0 – Setup (0.5 week)**

* Pick base 7–8B model (e.g., LLaMA‑3‑Instruct or Qwen‑2.5‑Instruct).
* Set up GPU/infra (4×A40 or 4090/5090).

**Phase 1 – Data + environment (1 week)**

* Load PrOntoQA (HF repo) and FOLIO/P‑FOLIO (HF + GitHub).
* Define option vocabulary from P‑FOLIO rule taxonomy.
* Implement NL→FOL mapping via existing FOLIO code and PrOntoQA generators.
* Implement solver wrapper: `check_step(state, option) → {valid, invalid, new_state}`.

**Phase 2 – Optionized SFT + constrained decoding (1 week)**

* Convert P‑FOLIO proofs and PrOntoQA traces into Thought/Action format.
* Train SFT model (LoRA) on this format to ensure syntax and basic semantics.
* Integrate constrained decoding (Outlines / SGLang / similar) for the `<Option …>` action segment.

**Phase 3 – OaK‑DPO loop (1–1.5 weeks)**

* For each dataset stage:

  * Sample k traces per problem from current model.
  * Run solver, label steps/traces.
  * Train (\hat{q}_\phi(s,\omega)) on all steps.
  * Build preference pairs and run DPO fine‑tuning.
* Run 2–3 outer iterations (PrOntoQA pretraining, then FOLIO/P‑FOLIO).

**Phase 4 – Evaluation + analysis (0.5–1 week)**

* Evaluate on held‑out FOLIO and OOD PrOntoQA.
* Compute accuracy, consistency, full‑trace validity, calibration curves.
* Run ablations (no constrained decoding; no option head; CoT vs optionized DPO).
* Produce key plots/tables for the paper.

This is tight but realistic on 4×A40 / 4090/5090, and it directly addresses novelty, workshop fit, implementability, and future extensibility.
