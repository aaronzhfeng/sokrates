# Changelog

All notable changes to the SOKRATES project.

## [0.1.1] - 2024-12-09

### Changed

#### Configuration Optimized for H100 PCIe
- Updated `configs/training.yaml` for single H100 PCIe (80GB):
  - SFT: `batch_size` 4→8, `gradient_accumulation_steps` 8→4
  - DPO: `batch_size` 2→4, `gradient_accumulation_steps` 16→8
  - OaK: `samples_per_problem` 8→4 (for faster training)
- Updated `configs/model.yaml`:
  - Added `attn_implementation: "flash_attention_2"` for H100
  - Added hardware documentation comments
- Added hardware estimation metadata to configs

#### Estimated Training Time
- Previous (4× A40): ~24-36 hours
- Current (1× H100 PCIe): ~3-4 hours
- Estimated cost: $7-10 @ $2.39/hr

---

## [0.1.0] - 2024-12-09

### Added

#### Core Infrastructure
- Project structure with `src/`, `configs/`, `scripts/`, `tests/`, `docs/`
- `pyproject.toml` with full package configuration
- `requirements.txt` with all dependencies

#### Data Module (`src/data/`)
- `structures.py`: Core data classes
  - `OptionType` enum with 18 inference rules
  - `FOLFormula` for first-order logic formulas
  - `ProofStep` for Thought/Action format steps
  - `LogicalState` for proof states
  - `OptionizedTrace` for complete traces
  - `PreferencePair` for DPO training
- `optionizer.py`: Proof optionization
  - P-FOLIO rule mapping
  - PrOntoQA conversion
  - Trace parsing

#### Models Module (`src/models/`)
- `option_head.py`: Option success predictor (q̂_φ)
  - `OptionSuccessHead` base implementation
  - `OptionSuccessHeadWithArgs` with argument encoding
- `gvf_heads.py`: General Value Function heads
  - `ConsistencyGVF` for contradiction-free prediction
  - `GoalProgressGVF` for goal proximity
  - `ProofLengthGVF` for remaining steps
  - `CombinedGVFHead` with shared features

#### Solvers Module (`src/solvers/`)
- `base_solver.py`: Abstract solver interface
  - `FOLSolver` ABC
  - `VerificationResult` dataclass
  - `ValidityStatus` enum
- `folio_solver.py`: Z3-based FOLIO solver
  - Step verification for major inference rules
  - Entailment checking
  - Consistency checking
- `prontoqa_solver.py`: Ontology-based PrOntoQA solver
  - Context parsing
  - Category derivation via forward chaining
  - Query verification

#### Training Module (`src/training/`)
- `sft.py`: Supervised Fine-Tuning
  - `SFTConfig` configuration class
  - `OptionizedTraceDataset` for training
  - LoRA/PEFT integration
  - Complete training pipeline
- `dpo.py`: Direct Preference Optimization
  - `DPOConfig` configuration class
  - Preference pair construction
  - TRL DPOTrainer integration
- `oak_loop.py`: OaK training loop
  - `OaKLoopConfig` configuration class
  - `OaKLoop` orchestration class
  - Iterative generate→verify→train cycle
  - Checkpoint saving

#### Inference Module (`src/inference/`)
- `constrained_decode.py`: Grammar-constrained decoding
  - `OptionConstrainer` for validation
  - Action string validation and fixing
  - EBNF grammar specification
- `generate_trace.py`: Trace generation
  - `GenerationConfig` settings
  - `TraceGenerator` class
  - Batch generation support

#### Evaluation Module (`src/evaluation/`)
- `metrics.py`: Evaluation metrics
  - Accuracy computation
  - Step validity metrics
  - Trace validity metrics
  - Brier score and ECE
  - Calibration curves
- `calibration.py`: Calibration analysis
  - `CalibrationAnalyzer` class
  - Per-option breakdown
  - Save/load functionality

#### Scripts
- `prepare_data.py`: Data download and preparation
- `train_sft.py`: SFT training script
- `run_oak_dpo.py`: OaK-DPO loop script
- `evaluate.py`: Model evaluation script

#### Configuration
- `configs/model.yaml`: Model and LoRA settings
- `configs/training.yaml`: SFT, DPO, OaK loop settings
- `configs/evaluation.yaml`: Evaluation and ablation settings

#### Tests
- `tests/test_structures.py`: Data structure tests
- `tests/test_solvers.py`: Solver tests
- `tests/test_metrics.py`: Metrics tests

#### Documentation
- `README.md`: Project overview and quick start
- `docs/00_index.md`: Documentation index
- `docs/05_technical_spec.md`: Technical specification
- `docs/06_glossary.md`: Terminology glossary
- `docs/07_implementation_guide.md`: Code walkthrough
- `docs/08_api_reference.md`: API documentation

### Technical Details

#### Option Vocabulary
- 18 inference rules from P-FOLIO taxonomy
- Argument count validation
- Human-readable descriptions

#### Model Architecture
- Base: 7-8B instruction-tuned LLM
- LoRA: r=64, alpha=128
- Option head: 64-dim embeddings, 512-dim MLP
- GVF heads: 256-dim shared, 64-dim per-task

#### Training Pipeline
- SFT: 3 epochs, lr=2e-5, batch=4×8 gradient accumulation
- DPO: 1 epoch, β=0.1, lr=5e-6
- OaK: 3 iterations, 8 samples per problem

#### Supported Datasets
- FOLIO (with FOL annotations)
- P-FOLIO (with step-level proofs)
- PrOntoQA (synthetic ontology reasoning)

### Dependencies
- PyTorch >= 2.1.0
- Transformers >= 4.36.0
- PEFT >= 0.7.0
- TRL >= 0.7.0
- Outlines >= 0.0.34
- Z3-solver >= 4.12.0

