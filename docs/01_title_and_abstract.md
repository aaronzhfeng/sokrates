**Title**

**SOKRATES: Distilling Symbolic Knowledge into Option-Level Reasoning via Solver-Guided Preference Optimization**

---

**Abstract**

Large language models (LLMs) frequently produce logically invalid chain-of-thought (CoT) reasoning even when their final answers are correct. Existing neuro-symbolic approaches, such as LoCo-LMs and solver-verified CoT methods, improve consistency by enforcing logical constraints or filtering proofs, but they generally treat reasoning as unstructured text and do not explicitly model reasoning actions or their predictive reliability. We introduce **SOKRATES** (Symbolic Option-Knowledge Reasoning Alignment via Trace Evaluation with Solver), a method that instantiates Sutton’s *Options and Knowledge* perspective in a first-order logic (FOL) micro-world.

SOKRATES represents proofs on benchmarks like FOLIO and P-FOLIO as sequences of discrete reasoning **options**—inference-rule macros such as `MODUS_PONENS` or `UNIV_INSTANTIATION` with arguments—rather than raw tokens. A first-order solver, available through FOL annotations and synthetic environments such as PrOntoQA, supplies ground-truth **knowledge** by verifying whether each option application is logically valid. From solver feedback, we (i) train an explicit option-success predictor (\hat{q}_\phi(s,\omega)) and (ii) construct preference pairs over full optionized traces, where solver-valid proofs are preferred to invalid ones. We then apply Direct Preference Optimization (DPO) to align a 7–8B LLM’s option policy with these solver-induced preferences.

Experiments on FOLIO and PrOntoQA show that SOKRATES improves final accuracy, full-trace logical validity, and calibration of (\hat{q}_\phi) compared to supervised fine-tuning, a semantic-loss baseline, and a solver-validated CoT baseline, yielding a concrete OaK-style loop for symbolic reasoning.
