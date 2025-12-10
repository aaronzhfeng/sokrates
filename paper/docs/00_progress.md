# SOKRATES Paper Progress Log

**Last Updated:** 2024-12-10 (Session 2)

---

## 📄 Paper Overview

| Field | Value |
|-------|-------|
| **Title** | SOKRATES: Distilling Symbolic Knowledge into Option-Level Reasoning via Solver-Guided Preference Optimization |
| **Venue** | AAAI 2026 (Anonymous Submission) |
| **Pages** | 7 (6 main + 1 appendix) |
| **Status** | Draft — awaiting experimental results |

---

## ✅ Completed

### Structure & Setup
- [x] AAAI 2026 template integrated (`aaai2026.sty`, `aaai2026.bst`)
- [x] Modular organization (figures/, tables/, algorithms/)
- [x] TinyTeX compilation pipeline working
- [x] Bibliography with 22 references (expanded)

### Sections Written
- [x] Abstract
- [x] §1 Introduction (3 contributions + architecture figure)
- [x] §2 Background and Related Work (restructured with 5 subsections)
  - 2.1 LLM Reasoning and Failure Modes
  - 2.2 Logical Reasoning Benchmarks
  - 2.3 Neuro-Symbolic Methods and Solver-Augmented Reasoning
  - 2.4 Preference Learning and Process Supervision
  - 2.5 Options, OaK, and Hierarchical RL
- [x] §3 Problem Setup: OaK in a Logic World
- [x] §4 Method: SOKRATES (5 subsections)
- [x] §5 Experimental Setup
- [x] §6 Results and Analysis (structure only)
- [x] §7 Ablation Studies (structure only)
- [x] §8 Conclusion
- [x] Appendix A: Prompt and Generation Details

### Figures Created
- [x] Figure 1: Architecture diagram (TikZ — in Introduction)
- [x] Figure 2: Example optionized proof trace
- [x] Figure 3: Full prompt template (Appendix)

### Tables Created
- [x] Table 1: Option vocabulary (complete)
- [x] Table 2: Main results (placeholder values)
- [x] Table 3: Ablations (placeholder values)

### Algorithm
- [x] Algorithm 1: SOKRATES Training Loop

---

## 🔲 Placeholders Awaiting Experiments

### Table 2: Main Results (`tables/main_results.tex`)
| Group | Model |
|-------|-------|
| *No Training* | Base CoT, Self-Consistency (k=8) |
| *Prior Methods* | LoGiPT, Logic-LM |
| *Preference Baselines* | Answer-only DPO, CoT-DPO, VeriCoT |
| *Ours* | SFT, SOKRATES (iter 1), SOKRATES (iter 2) |

**→ 10 rows × 4 metrics = 40 values needed**

### Table 3: Ablations (`tables/ablations.tex`)
| Group | Ablation |
|-------|----------|
| *Full* | SOKRATES (full) |
| *Representation* | w/o optionization, w/o Thought, w/o constrained dec. |
| *Knowledge* | w/o option head (q̂), w/o solver verification |
| *Iterations* | 1 iteration only, 3 iterations |
| *Sampling* | K=4 samples, K=8 samples |

**→ 10 rows × 3 metrics = 30 values needed**

### Narrative Sections
| Location | Content Needed |
|----------|----------------|
| §6.1 | Main results narrative |
| §6.2 | Calibration analysis results |
| §6.3 | FOLIO transfer results |

---

## 📊 Placeholder Summary

| Category | Count |
|----------|-------|
| Table 2 values | 40 |
| Table 3 values | 30 |
| Narrative sections | 3 |
| Figure diagrams | **0** ✅ |
| **Total** | **73** |

---

## 📝 Edit Log

### Session 2 — 2024-12-10

| Time | Change | Files |
|------|--------|-------|
| 00:30 | Added preference scoring formula (Eq. 7) to §4.4 | sokrates.tex |
| 00:30 | Added SFT/DPO teaching descriptions (§4.1, §4.5) | sokrates.tex |
| 00:30 | Added Symbol column to Table 1 (options) | tables/options.tex |
| 00:20 | Expanded Table 2 (10 rows) and Table 3 (10 rows) with grouped baselines | tables/*.tex |
| 00:20 | Paper now 7 pages, 283KB | sokrates.pdf |
| 00:09 | **Major restructure per 03_improvement.md** | sokrates.tex, references.bib |
| 00:09 | — Restructured §2 into 5 subsections (LLM failures, benchmarks, neuro-symbolic, preferences, OaK) | |
| 00:09 | — Moved architecture figure to end of Introduction | |
| 00:09 | — Added 6 new references (Self-Consistency, Cannot Self-Correct, RuleTaker, ToT, BoT, VeriCoT) | |
| 00:09 | — Added explicit VeriCoT comparison in §2.4 | |
| 00:09 | — Expanded OaK/reward-respecting discussion in §2.5 | |
| 00:09 | — Paper now 274KB | sokrates.pdf |
| 23:30 | Fixed naming: "OaK-DPO" → "SOKRATES loop" throughout | sokrates.tex, 02_improvement.md |
| 23:25 | Redesigned trace example figure (cleaner layout, TikZ boxes) | figures/trace_example.tex |
| 23:15 | Created TikZ architecture diagram (Figure 1) | figures/architecture.tex |
| 23:15 | Installed pgf package for TikZ support | system |
| 22:46 | Installed TinyTeX for proper pdflatex with bold fonts | system |
| 22:50 | Fixed bold title/headers (was using XeTeX, now pdflatex) | sokrates.tex |
| 22:55 | Modularized paper into figures/, tables/, algorithms/ | sokrates.tex, 6 new files |
| 23:00 | Added improvements from 02_improvement.md | sokrates.tex |
| 23:02 | Created figures/full_prompt.tex | figures/full_prompt.tex |

### Session 1 — 2024-12-09

| Time | Change | Files |
|------|--------|-------|
| -- | Initial paper backbone created | sokrates.tex |
| -- | AAAI template files added | aaai2026.sty, aaai2026.bst |
| -- | References populated | references.bib |
| -- | All sections drafted | sokrates.tex |

---

## 🔧 Build Instructions

```bash
# Set PATH for TinyTeX
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

# Full build
cd /raid/zhf004/sokrates/paper
pdflatex sokrates.tex
bibtex sokrates
pdflatex sokrates.tex
pdflatex sokrates.tex

# Or use Makefile
make clean && make
```

---

## 📁 File Structure

```
paper/
├── sokrates.tex          # Main document
├── sokrates.pdf          # Compiled output (274KB, 6 pages)
├── references.bib        # Bibliography (22 entries)
├── aaai2026.sty          # AAAI style file
├── aaai2026.bst          # AAAI bibliography style
├── Makefile              # Build automation
├── docs/
│   ├── 00_progress.md    # This file
│   ├── 01_improvement.md # Content improvement notes
│   └── 03_improvement.md # Structure improvement notes
├── figures/
│   ├── architecture.tex  # Fig 1: Architecture (TikZ)
│   ├── trace_example.tex # Fig 2: Example proof trace
│   └── full_prompt.tex   # Fig 3: Prompt template (appendix)
├── tables/
│   ├── options.tex       # Tab 1: Option vocabulary
│   ├── main_results.tex  # Tab 2: Main results (placeholders)
│   └── ablations.tex     # Tab 3: Ablations (placeholders)
└── algorithms/
    └── oak_loop.tex      # Alg 1: Training loop
```

---

## 📚 References Added (Session 2)

| Citation | Topic |
|----------|-------|
| `wang2023selfconsistency` | Self-consistency decoding |
| `huang2024large` | LLMs cannot self-correct reasoning |
| `clark2021transformers` | RuleTaker / ProofWriter |
| `yao2024tree` | Tree of Thoughts |
| `yang2024buffer` | Buffer of Thoughts |
| `ling2023deductive` | VeriCoT (key comparison) |

---

## 🎯 Next Steps

1. **Run experiments** to get actual numbers
2. **Fill Table 2** with main results
3. **Fill Table 3** with ablation results  
4. **Write result narratives** (§6.1, §6.2, §6.3)
5. ~~Create architecture diagram~~ ✅ Done
6. ~~Restructure related work~~ ✅ Done
7. **Final polish** and submission
