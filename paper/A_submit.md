# SOKRATES Submission Summary

Venue: AAAI-40 Bridge Program on Logical and Symbolic Reasoning in Language Models  
Date: January 20-21, 2026, Singapore EXPO

---

## Title

SOKRATES: Distilling Symbolic Knowledge into Option-Level Reasoning via Solver-Guided Preference Optimization

---

## Authors

Zhaoxiang Feng¹, David Scott Lewis²

¹University of California San Diego, USA  
²AI Executive Consulting (AIXC), Zaragoza, Spain

Emails: zhf004@ucsd.edu, reports@aiexecutiveconsulting.com

---

## Abstract

A language model that achieves 94% accuracy on logical reasoning sounds impressive—until you discover that only 2% of its proofs are actually valid. This is the state of chain-of-thought prompting: models produce plausible rationales that frequently contain invalid inference steps, hidden contradictions, or skipped derivations. The right answer emerges despite, not because of, the reasoning process. We introduce SOKRATES (Symbolic Option-Knowledge Reasoning Alignment via Trace Evaluation with Solver), a method that instantiates Sutton's Options and Knowledge (OaK) framework in a first-order logic micro-world. SOKRATES represents proofs as sequences of discrete inference-rule options (e.g., MODUS_PONENS, UNIV_INSTANTIATION), verified step-by-step by a FOL solver. From solver feedback we (i) train an option-success predictor that estimates validity before execution, and (ii) construct preference pairs for Direct Preference Optimization (DPO), aligning the model's option policy with solver-induced correctness. On PrOntoQA, SOKRATES raises accuracy from 94.2% to 97.6%, step validity from 27.3% to 98.5%, and full-trace validity from 2.1% to 92.0%—a 33x improvement in logically sound proofs. The learned predictor is well calibrated (ECE = 0.08), and the option policy transfers zero-shot to FOLIO, improving accuracy from 45.3% to 53.2%. To our knowledge, SOKRATES is the first closed-loop OaK instantiation in a neural language model, demonstrating that the options-and-knowledge paradigm yields substantial empirical gains in symbolic reasoning.

---

## Keywords

Chain-of-thought reasoning, Logical reasoning, Neuro-symbolic AI, Direct Preference Optimization, Options and Knowledge framework, First-order logic, Solver-guided supervision, Process supervision, Proof verification

---

## TL;DR

LLMs get right answers via wrong reasoning (94% acc, 2% valid proofs). SOKRATES uses solver feedback to supervise inference-rule options, achieving 33x improvement in proof validity (2%→92%). First OaK loop in a neural LM.

---

## Key Results (PrOntoQA)

| Model | Accuracy | Step Validity | Trace Validity |
|-------|----------|---------------|----------------|
| SFT baseline | 94.2% | 27.3% | 2.1% |
| SOKRATES (iter 2) | 97.6% | 98.5% | 92.0% |

---

## Relevant Workshop Topics

- Chain-of-thought reasoning of LLMs
- External tool-use (logic solvers) for LLM reasoning
- Logical consistency of LLMs
- Symbolic expressions and reasoning of LLMs
- Benchmarks and evaluation for logical reasoning of LLMs
