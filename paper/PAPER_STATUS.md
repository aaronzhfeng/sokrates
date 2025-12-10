# SOKRATES Paper Status

**Target**: AAAI-26 Bridge Workshop on Logical and Symbolic Reasoning in Language Models

## Paper Files

```
paper/
├── sokrates.tex          # Main paper (~524 lines)
├── references.bib        # Bibliography (~17 references)
├── aaai2026.sty          # AAAI style file
├── aaai2026.bst          # AAAI bibliography style
├── figures/              # Figure directory (empty - needs architecture diagram)
├── Makefile              # Build automation
└── sokrates-paper.zip    # Ready for Overleaf upload
```

## Section Status

| Section | Status | Notes |
|---------|--------|-------|
| **Abstract** | ✅ Complete | ~180 words |
| **1. Introduction** | ✅ Complete | Motivation, problem, 3 contributions |
| **2. Background** | ✅ Complete | OaK, PrOntoQA/FOLIO, DPO |
| **3. Problem Setup** | ✅ Complete | States, Options, Knowledge, Thought/Action format |
| **4. Method** | ✅ Complete | Full pipeline description + Algorithm 1 |
| **5. Experimental Setup** | ✅ Complete | Datasets, baselines, metrics defined |
| **6. Results** | ⏳ Placeholder | Table 1 - waiting for experiments |
| **7. Ablations** | ⏳ Placeholder | Table 2 - waiting for experiments |
| **8. Related Work** | ✅ Complete | LoCo-LMs, Logic-LM, LINC, DPO, Options |
| **9. Conclusion** | ✅ Complete | Summary + limitations + future work |

## Figures Needed

1. **Architecture Diagram** (Figure 1)
   - Shows: Datasets → Optionizer → SFT → OaK Loop
   - Currently: Placeholder box in tex file
   - TODO: Create with TikZ or external tool

2. **Calibration Curves** (optional, for Results)
   - Shows: $\hat{q}_\phi$ predictions vs. actual validity
   - Depends on: Experiment results

## Tables Status

| Table | Status | Content |
|-------|--------|---------|
| Table 1 (Options) | ✅ Complete | Option vocabulary (11 options) |
| Table 2 (Main Results) | ⏳ Placeholder | Acc, Step Val, Trace Val, Brier |
| Table 3 (Ablations) | ⏳ Placeholder | Component analysis |

## Key References (included)

- Chain-of-Thought (Wei et al., 2022)
- PrOntoQA (Saparov & He, 2023)
- FOLIO (Han et al., 2022)
- DPO (Rafailov et al., 2023)
- Options Framework (Sutton et al., 1999)
- OaK / Reward-Respecting Subtasks (Sutton et al., 2023)
- LoCo-LMs (Riegel et al., 2020)
- Logic-LM (Pan et al., 2023)
- LINC (Olausson et al., 2023)
- LoRA (Hu et al., 2022)
- ReAct (Yao et al., 2023)
- Qwen2 (Yang et al., 2024)
- LoGiPT (Feng et al., 2024)

## Compilation

### Local (if LaTeX installed)
```bash
cd paper
make          # Builds sokrates.pdf
make clean    # Removes temp files
```

### Overleaf
1. Upload `sokrates-paper.zip` to Overleaf
2. Set compiler to pdfLaTeX
3. Compile

## Next Steps

1. **Run experiments** - Fix the device placement bug and run SOKRATES
2. **Fill in results** - Update Tables 2, 3 with actual numbers
3. **Create architecture figure** - Replace placeholder with real diagram
4. **Add calibration analysis** - If results show good $\hat{q}_\phi$ calibration
5. **Final polish** - Check references, formatting, page limit

## Page Estimate

Current draft is approximately 6-7 pages (AAAI format).
Workshop papers typically 4-8 pages.

