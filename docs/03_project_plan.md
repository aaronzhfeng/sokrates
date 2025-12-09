I’ll treat this as designing the *actual project* you could implement, not a hand-wavy idea.

---

## 1. Project in one sentence

Learn an **option-level reasoning policy** for FOL tasks where:

* each proof step is a **discrete option** (inference macro),
* a **FOL solver** provides ground-truth “knowledge” about whether options are valid,
* and **Direct Preference Optimization (DPO)** distills this knowledge into the LM so that it prefers solver‑consistent option sequences.

This is a small, concrete instantiation of Sutton’s **Options and Knowledge (OaK)** architecture in a logical micro-world, on top of FOLIO/P‑FOLIO/PrOntoQA.

---

## 2. Novelty and positioning

You have to differentiate from two very close lines of work:

* **LoCo‑LMs**: fine-tune with a neuro‑symbolic *semantic loss* so LMs respect logical constraints; they show improved factuality and logical self‑consistency.
* **VeriCoT**: parse CoT into FOL, verify each step with a solver, and use this for supervision and preference training; they already use solver‑based validation to improve CoT.

Your project needs **three differentiators**:

1. **Explicit options, not generic CoT**

   * Use **P‑FOLIO**’s step labels (inference rules) to define a *finite option vocabulary* and represent proofs as sequences of option tokens + arguments, not free-form text.
   * This is much closer to OaK: options are reusable skills.

2. **Explicit option models (knowledge), not just a better policy**

   * Train a small head (\hat{q}_\phi(s,\omega)) that predicts the probability an option (\omega) will be **solver‑valid** in state (s).
   * Measure calibration and use it in analysis as “OaK‑style knowledge” about options.

3. **Micro OaK loop, not pure one-shot fine‑tuning**

   * Run at least 2–3 **generate → verify → re‑train** cycles, so there is genuine “experience → updated options/models → improved behaviour.”
   * It’s still offline/batched, but it’s structure‑wise a small OaK/STOMP cycle.

That’s enough novelty for a bridge workshop if you’re honest about what you *aren’t* doing (no lifelong RL, limited feature discovery).

---

## 3. Method design

### 3.1 World, states, and options

**World / tasks**

* Use **FOLIO**: NL stories + FOL annotations, checked by an FOL engine.
* Use **P‑FOLIO**: human proof chains for a subset of FOLIO, with step‑by‑step NL proofs and inference rule tags.
* Use **PrOntoQA** as an out‑of‑distribution testbed (synthetic ontology‑based FOL questions with proofs).

**State representation**

* Logical state (s) for each problem =

  * NL story text,
  * set of currently derived FOL formulas (from annotations / previous steps),
  * target conclusion.

**Options**

* From P‑FOLIO labels, define a compact option set:

  * e.g., 10–20 macros: `UNIV_INSTANTIATION`, `EXIST_ELIM`, `MODUS_PONENS`, `AND_INTRO`, `AND_ELIM`, `OR_INTRO`, `CASE_SPLIT`, etc.
* Represent each proof step as:

  * an **option type** ID,
  * its arguments (which formulas/variables it applies to),
  * plus a natural‑language justification.

This is your option space. You train the LM to generate **optionized CoT** instead of raw text.

### 3.2 Knowledge: solver and option models

**External knowledge (ground truth)**

* FOLIO gives FOL formulas and uses a standard FOL inference engine to ensure correctness.
* Implement a wrapper:

  * takes current FOL facts + a candidate option application,
  * checks whether the new formula is entailed (e.g. by adding it and checking consistency / derivability),
  * returns VALID / INVALID and updated fact set.

**Internal knowledge (learned)**

* Train (\hat{q}_\phi(s,\omega)) on your logged steps:

  * Inputs: LM hidden state for that step (or pooled representation of state) + option type embedding.
  * Target: step validity (solver label).
  * Output: probability the step is valid.
* Also track empirical **success rates per option type** before/after training; these are crude option models.

This is exactly OaK’s “knowledge = models of options in terms of their transitions/success,” just reduced to step validity probability in a logic world.

### 3.3 Micro OaK loop with DPO

Define one outer **OaK iteration** as:

1. **Policy (\pi) (LM) generates experience**

   * For each training problem: sample (k) optionized CoTs from the current LM.

2. **Knowledge source (solver) labels experience**

   * For each step: VALID/INVALID.
   * For each full trace: VALID if all steps valid *and* conclusion matches label; else INVALID.

3. **Update option models (q_\phi)**

   * Train/update (\hat{q}_\phi(s,\omega)) on all steps; this is your explicit knowledge.

4. **Build preference pairs for DPO**

   * For each problem:

     * choose at least one VALID trace (if any) as winner,
     * choose one INVALID trace as loser.
   * If no valid trace exists, skip or create partial preferences (more valid steps > fewer).

5. **Policy improvement via DPO**

   * Use standard DPO objective to fine‑tune (\pi) so that it prefers VALID traces over INVALID ones, conditioned on the same prompt.

Repeat this outer loop a few times. That gives you a small but real **“experience → model → policy”** cycle in the OaK spirit, albeit offline.

### 3.4 Reward‑respecting subtasks (minimal version)

To nod more directly at **reward‑respecting subtasks** and GVFs in OaK/STOMP :

Define 1–2 simple subtasks:

1. **Consistency subtask**

   * Reward = 1 at the end if *no* step was invalid (no contradictions / illegal inferences), 0 otherwise.
   * Learn a small head (v_{\psi}^{\text{consistency}}(s)) that predicts this reward from the LM state at each step (off‑policy from logs).

2. **Goal‑progress subtask**

   * Reward = 1 if a step derives a formula closer to the conclusion (e.g., introduces the right predicate), 0 otherwise.
   * Learn (v_{\psi}^{\text{goal}}(s)) similarly.

You don’t use these GVFs to control yet; they serve as **explicit predictive knowledge** you can report, and they mirror the STOMP/OaK vocabulary of “feature attainment” and “reward‑respecting subtasks.”

---

## 4. Experimental design

### 4.1 Datasets

* **Train / dev:**

  * **P‑FOLIO**: step‑annotated proofs to train SFT and define options.
  * Subset of **FOLIO** without proofs for generation and verification.

* **Test / generalization:**

  * Held‑out FOLIO cases.
  * **PrOntoQA** synthetic worlds and OOD settings (different ontology distributions).

Optionally, one LogicBench subset for diversity.

### 4.2 Models and baselines

Base model: 7–8B open LLaMA/Qwen‑family instruct model; LoRA/PEFT for fine‑tuning.

Compare:

1. **Base**: zero‑shot / few‑shot CoT.
2. **SFT**: supervised on P‑FOLIO proofs (optionized CoT).
3. **LoCo‑style semantic loss**: reimplement simplified version of LoCo‑LMs on a subset (semantic loss enforcing FOL constraints).
4. **VeriCoT‑style baseline**: solver‑verified CoT + SFT/DPO, *without* explicit options/option‑models.
5. **OaK‑DPO**: your full method (options + option models + DPO loop).

### 4.3 Metrics

* **Task level**

  * Final answer accuracy (correct yes/no/label).
* **Proof level**

  * Step validity rate (% of steps solver‑valid).
  * Full-trace validity (% of proofs where all steps valid *and* answer correct).
* **Knowledge level**

  * Calibration of (\hat{q}_\phi(s,\omega)): Brier score / ECE for predicting validity.
  * For subtasks: calibration of (v_{\psi}^{\text{consistency}}, v_{\psi}^{\text{goal}}).
* **Ablation metrics**

  * Effect of options vs raw CoT on DPO outcome.
  * Effect of solver‑based preferences vs final‑answer-only preferences.

This directly targets the workshop’s interest in **avoiding contradictions**, logical QA, and symbolic integration.

---

## 5. Implementation feasibility and compute

Given 4×A40 / 4090/5090, this is realistic:

* **Data scale**

  * FOLIO is ~1.4k examples; P‑FOLIO is of similar order.
  * You can afford to generate multiple CoTs per example (e.g., k=8–16) → tens of thousands of traces.

* **Training**

  * SFT on proofs: a few hours on 1–2 GPUs for 7–8B with LoRA.
  * DPO on ~50–100k preference pairs: similar or less. DPO is designed to be lightweight compared to PPO‑style RLHF.
  * Option-model and GVF heads: cheap supervised heads piggybacking on existing traces.

* **Symbolic side**

  * FOLIO’s FOL annotations and inference engine already exist; you’re not writing a prover from scratch.
  * PrOntoQA world models are synthetic and clean.

This is well within your hardware; bottleneck is careful engineering, not FLOPs.

---

## 6. Future work trajectories (you can flag these explicitly in the paper)

This is where you sell the “architecture” story to the workshop:

1. **From micro‑OaK to continual OaK**

   * Move from batched iterations to genuine online learning: run the OaK loop while new logical tasks arrive, with non‑stationary distributions.

2. **Richer option models**

   * Extend (\hat{q}_\phi) and GVFs to predict not just step validity, but distributions over *types of future states*, expected proof length, etc.

3. **Feature discovery**

   * Borrow ideas from OaK’s feature‑construction step to automatically propose new logical “features” (e.g., useful lemmas or invariants) that spawn new subtasks and options.

4. **Temporal / multi‑timescale reasoning**

   * Add temporal logic datasets and multi‑horizon options; plug into NeSy temporal frameworks later.

5. **Beyond FOL micro‑worlds**

   * Apply the same OaK‑DPO idea to math proofs, program verification, or planning tasks (PDDL), where options are tactics or planning macros and knowledge comes from domain solvers.

---

## 7. Literature anchors (for your related‑work and positioning)

* **OaK / STOMP / reward‑respecting subtasks**

  * Sutton, *Reward‑Respecting Subtasks for Model‑Based RL* and the “Alberta Plan” slides: features → subtasks → options → models → planning.

* **Logical reasoning benchmarks**

  * FOLIO (NL + FOL annotations checked by an FOL engine).
  * P‑FOLIO (NL proofs + rule labels).
  * PrOntoQA and LogicBench (synthetic FOL / ontology reasoning).

* **Neuro‑symbolic consistency & solver‑guided CoT**

  * LoCo‑LMs (semantic loss).
  * VeriCoT (formalizing CoT steps in FOL + solver verification and feedback).

* **Alignment / DPO**

  * Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (closed‑form RLHF without RL).

* **Workshop context**

  * AAAI‑26 Bridge “Logical and Symbolic Reasoning in Language Models”: explicitly about using logic and symbolic methods to enhance LLM reasoning and consistency.

This design hits all six constraints: nontrivial novelty (optionization + option models + micro‑OaK loop), directly on topic, very implementable within your GPU budget, and with obvious extension paths into full OaK and broader NeSy reasoning.
