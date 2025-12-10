# DPO/OaK Runtime Notes (Dec 2025)

Context: Hotfixes applied while a multi-GPU OaK-DPO run was active. Changes target data consistency across ranks, config immutability, and safe prompt stripping for DPO pairs.

## Changes Implemented
- **Distributed trace gather**: `src/training/oak_loop.py` now initializes `torch.distributed` when `WORLD_SIZE>1` and gathers generated traces across ranks before preference construction. Prevents per-rank dataset skew and DDP deadlocks when `DPOTrainer` starts.
- **Immutable per-iteration DPO config**: `src/training/dpo.py` clones the base `DPOConfig` per iteration (using `replace`) and writes checkpoints to `.../iter_{k}` without mutating the shared config. Avoids nested `iter_*` paths and cross-iteration side effects.
- **Safe prompt stripping in pairs**: `src/data/structures.py` removes the prompt prefix exactly once (prefix check + slice) instead of blanket `.replace`, preventing accidental deletion when prompt text appears inside generated reasoning.

## Operational Guidance
- Launch with `torchrun`/`accelerate` as before; no new CLI flags. The loop will self-init dist if `WORLD_SIZE>1`. NCCL env (e.g., `NCCL_DEBUG=INFO`, `NCCL_P2P_DISABLE` for PCIe-only) still recommended.
- DPO checkpoints now live under `<base_output>/dpo/iter_{k}` (or `oak_loop` equivalent). Expect one level of `iter_*`, not nested.
- If running single-GPU (`WORLD_SIZE=1`), behavior is unchanged; dist is not initialized.
- Preference pair text now retains any prompt-like content that appears later in the trace. Downstream DPO training should see cleaner `prompt`/`chosen`/`rejected` separation.

## Validation Recommendations
- Smoke test distributed path: `torchrun --nproc_per_node=2 scripts/run_oak_dpo.py --samples 1 --iterations 1 --config configs/training.yaml --train-data <small_split> --sft-model <ckpt> --dataset-type prontoqa`.
- Confirm logs show `Distributing ... problems across ... GPUs` followed by `Generated ... total traces` (identical across ranks).
- Inspect `outputs/.../dpo/iter_0` (or current run) for checkpoint layout and non-empty `trainer_state.json`.
- Spot-check a few serialized preference pairs to ensure `prompt` is intact and `chosen`/`rejected` contain reasoning without dropped text.

## Residual Risks / Notes
- Dist init assumes NCCL backend; if running on CPU-only or non-NCCL-capable interconnect, set `WORLD_SIZE=1` or override backend before launch.
- Long-running jobs started before this change will keep their prior behavior; restart to pick up the fixes.

