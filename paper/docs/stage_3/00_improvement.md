## What the paper currently does (and why it works)

The paper’s core move is to stop treating reasoning as free-form text and instead treat it as a *sequence of discrete inference-rule actions* (“options”), each checked step-by-step by a FOL solver. From solver feedback it trains (i) an option-success predictor ( \hat q_\phi(s,\omega) ) and (ii) an option policy aligned with solver-validated traces via solver-derived DPO preferences.  

Empirically, the headline result is very strong: SFT gets high accuracy but extremely low step/trace validity (94.2% acc vs 2.1% fully valid traces), while Sokrates iter-2 reaches 97.6% acc and 92.0% fully valid traces.  This directly supports the paper’s motivating claim about “right answer, wrong reasoning.” 

## Strengths to preserve

* **Clear task framing + formalization:** State (s=(P,D,c)) and options as inference macros is clean and readable.  
* **Solver signal used in two ways:** both as labels for ( \hat q_\phi ) and to build DPO preference pairs.  
* **Calibration angle is valuable:** you explicitly measure ECE/Brier and argue why calibration matters for search/planning. 
* **Error analysis is specific:** premise misidentification / rule misuse / premature conclude are concrete and actionable. 

---

## What’s already improved in the latest version (Dec 2025)

This note predates a formatting + consistency cleanup. As of the latest paper revision:

* **AAAI/AuthorKit compliance pass (DONE):**
  * No `Overfull \hbox` margin violations in the LaTeX log.
  * Table captions moved to appear **under** tables (per AuthorKit).
  * Removed negative spacing hacks near tables/figures/captions.
  * Table font sizes kept within the allowed range (avoid `\scriptsize` tables).

* **Internal consistency fixes (DONE):**
  * Option/rule naming aligned across prompt, tables, trace example, and paper text.
  * Hyperparameter statements aligned (e.g., \(K\), \(T_{\max}\), greedy vs \(\tau>0\)).
  * Algorithm pseudocode no longer hard-codes mismatched values (e.g., `K=8`).

* **Presentation polish (DONE):**
  * DPO objective rewritten into an equivalent one-line form to avoid awkward line breaks.
  * OaK mapping table reformatted to avoid stretched words (ragged-right columns + more row spacing).

For a full change log, see `paper/docs/stage_2/02_conversation_progress_2025-12-12.md`.

---

## High-impact ways to improve the paper

### 1) Fill in “missing mechanics” so the method is reproducible

Right now, several implementation-critical parts are implied but not specified enough for replication:

* **Which verifier is used (and how strict is it), per dataset?** The paper defines the solver interface abstractly, but reviewers will expect explicit details because “step validity” is a core metric.
  **Fix:** add a short “Verifier details” paragraph including:
  - verifier identity (e.g., Z3 vs ontology-based checker), version, and timeout,
  - how each option is encoded/checked (especially quantifier rules),
  - what happens when a step cannot be parsed or verified (reject vs accept vs fallback).

* **Define the state update precisely.** You say the model “parse[s] option ( \omega_t ), update state (s_{t+1})”  but the exact rules for indexing (premises vs derived) and how derived formulas are appended determine whether “premise misidentification” is a modeling failure or a bookkeeping artifact.
  **Fix:** include 6–10 lines of pseudocode: given action args ([i,j]), map to (premise/derived) objects, compute derived formula string/FOL, append to (D), renumber.

* **Clarify what gets verified on CONCLUDE.** Table 2 defines CONCLUDE as terminal with label encoding.  But solver-validity for CONCLUDE is ambiguous: do you require the conclusion formula to be present in (D)? do you check entailment (P \cup D \models c)?
  **Fix:** explicitly state the acceptance criterion for CONCLUDE and whether “premature conclusion” is defined as failing that criterion (your error analysis suggests it is). 

### 2) Strengthen the preference-learning design (your current scoring is a weak point)

Your preference score mixes (a) step-validity rate, (b) correctness, and (c) a bonus for fully valid traces.  This is plausible, but it’s also arbitrary and may be doing accidental shaping.

Concrete improvements:

* **Make preferences lexicographic instead of additive.** If your real objective is “sound proof first, then correctness,” you should enforce it structurally:

  * prefer any (V(\tau)=1) over any (V(\tau)=0)
  * among (V(\tau)=1), prefer correct over incorrect
  * among ties, prefer higher step-validity or shorter proofs
    This directly targets the “premature conclusion” mode. 

* **Ablate the weighting.** Right now the +0.5 for fully valid traces is unmotivated. 
  **Add:** a sensitivity plot over that weight (0, 0.5, 1, 2, “hard constraint”) showing trace validity/accuracy. This is a small experiment but plugs a reviewer-shaped hole.

* **Use solver error types to build “cleaner” losers.** You already see dominant error types (wrong indices, wrong rule, premature conclude). 
  **Add:** construct preference pairs where winner/loser differ in *one factor* (e.g., same rule but wrong indices) to reduce confounding and make learning signal sharper.

### 3) Use ( \hat q_\phi ) for *inference-time* gains (otherwise it looks underused)

You explicitly list multiple test-time uses of calibration (backtracking, best-of-K scoring, pruning).  But the current system uses ( \hat q_\phi ) only during training, and you list that as a limitation. 

This is low-hanging fruit and would strengthen the paper substantially:

* **Add a single test-time algorithm:** “best-of-K traces ranked by (\prod_t \hat q_\phi(s_{t-1},\omega_t))” (you already suggest it). 
  Evaluate: accuracy, trace validity, and solver calls (if you can prune). Even a modest gain is valuable because it demonstrates ( \hat q_\phi ) is not just an auxiliary metric.

* **Add a lightweight “validity-gated decoding” variant:** at each step, sample M candidate actions, score by ( \hat q_\phi ), take max. This avoids complex tree search but still shows planning value.

### 4) Expand baselines and ablations to preempt predictable reviewer pushback

Right now, baselines are mainly “Base CoT”, “Self-consistency”, and “SFT.”  Given related work you cite, reviewers will ask: “How does this compare to VeriCoT / Logic-LM-style verification preferences?”

You can address this without reimplementing everything:

* **Add a *closest-possible* baseline:**
  “Verifier-based DPO on unstructured CoT” (even if your parser is crude) as a proxy for VeriCoT-style training. Then show your optionized approach beats it on trace validity and/or transfer. You already motivate that unstructured CoT is weaker. 

* **Ablate each component explicitly:**

  1. constrained decoding on Action vs unconstrained (you claim grammar removes syntax errors) 
  2. DPO w/ solver-step signal vs answer-only (you have this) 
  3. DPO with/without the ( \hat q_\phi ) auxiliary head (currently missing)
  4. removing Thought generation entirely (Action-only traces). If performance holds, you can argue the framework is about actions, not chain-of-thought.

### 5) Address the “K=2” concern more convincingly

You state K=2 is due to compute and mention “full design uses K=8.”  Reviewers will interpret “K=8 would be better but we didn’t do it” as unfinished.

What to do:

* You already have a small hyperparameter table showing K=4 changes diversity. 
  **Add one more result:** run 1 iteration with K=4 on a subset (e.g., 2k problems) and report trace validity gains per solver call. This reframes compute as an explicit tradeoff curve.

* **Report the actual number of solver calls.** Since solver bottleneck is a limitation , quantify it: calls per training example, average proof length, average solver time.

### 6) Improve transfer story: explain *why* transfer is limited and how to extend

You show zero-shot transfer from PrOntoQA-trained Sokrates to FOLIO improves accuracy (45.3 → 53.2) and trace validity (9.9 → 14.8).  This is good, but still low validity overall, and you should control the narrative.

Concrete improvements:

* **Diagnose transfer failures with the same taxonomy.** Are FOLIO errors still mostly misindexing, or is semantic parsing the bottleneck? You note FOLIO is natural language and more varied. 
  Add: a 30-example annotated analysis on FOLIO so the paper doesn’t look like “transfer works a bit, unexplained.”

* **Option vocabulary mismatch:** your option set is tailored to PrOntoQA patterns. 
  Add: a coverage stat on FOLIO (“% of gold proofs expressible with our 11 rules”), or at least a qualitative note of missing rule types (e.g., contraposition variants, existential instantiation, equality reasoning).

### 7) Fix writing/formatting issues that will cost you points for free

These are small but reviewers notice them:

* Model citation mismatch risk: you say “Qwen3-8B” but cite “Qwen2 technical report.”  This looks sloppy even if the underlying model choice is fine.
* Consider tightening the “options are temporally extended” justification: in this setup each step is one inference-rule application, so calling them temporally extended options will trigger nitpicks. Your argument is that each “option” bundles sub-choices (rule + indices + justification + state update).  Make that explicit earlier to defuse the objection.
* Camera-ready workflow nit: AAAI AuthorKit often requires a **single `.tex`** source file (no `\input`) for the final archive. If you keep modular `\input`s for development, document how to produce a single-file build for submission.

---

## Method-level ideas to reduce your main remaining errors (based on your own error analysis)

You identify: premise misidentification (42%), incorrect rule application (31%), premature conclusion (27%).  Here are targeted fixes:

1. **Premise misidentification**

* Add a *typed premise header* in the prompt: for each premise/derived formula, include a cheap parser tag like `{IMPL}`, `{FORALL}`, `{DISJ}`, `{CONJ}`, `{NEGNEG}`. This makes “UI requires ∀” easier. Your state already includes premises and derived formulas. 
* Split action generation into two heads: rule-type first, then indices (pointer-style). This reduces the “one token string decides everything” issue.

2. **Incorrect rule application**

* During decoding, enumerate only *applicable* rules given the current formula tags (syntactic applicability), then let the model choose among them. You already separate syntax errors (grammar) from semantic errors (solver).  This adds a middle layer: “syntactically applicable” vs “logically valid.”

3. **Premature conclusion**

* Hard constraint: CONCLUDE is only allowed if conclusion (c) is present in derived set (D), or solver can prove (P \cup D \models c). This aligns directly with your “trace validity” definition. 
* Or, if you want to allow early stopping, redefine validity to permit “entailed without explicit derivation” — but then you must change the metric and explain it.

---

## Quick checklist of edits that would most improve acceptance odds

* **(DONE)** Add verifier identity + encoding details + CONCLUDE verification rule.
* **(TODO)** Preference learning: ablate the +0.5 fully-valid bonus and/or switch to lexicographic preferences.
* **(TODO)** Add *one* inference-time use of \(\hat q_\phi\) (best-of-\(K\) reranking is easiest).
* **(TODO)** Add missing ablations: remove Thoughts; remove constrained decoding; remove \(\hat q_\phi\).
* **(DONE)** Fix the model citation mismatch (Qwen3 vs Qwen2 report).
* **(DONE)** Formatting compliance: captions under tables, no overfull boxes, no negative spacing near floats.
