# Code-to-Reasoning Transfer Experiments

Clean, code-only release for the experiments reported in **Why Does Code
Improve Reasoning? On Code-to-Reasoning Transfer in LLM Post-Training Through
a Decompositional Lens**.

This repository was reconstructed from the active project snapshot and the
read-only AOSS archive. It intentionally excludes datasets, model weights,
checkpoints, raw generations, logs, and large result files. Small aggregate
aggregate reference packages are included so the main analysis and Appendices
A, B and D can be regenerated without rerunning hundreds of GPU jobs.

## Experimental stages

| Stage | Paper role | Method | Data family |
|---|---|---|---|
| 01 | Main text | 10-category matched LoRA-SFT ablation | cleaned KodCode, 35,974 train rows |
| 02 | Appendix B | 0.5/1.0-epoch LoRA-SFT diagnosis | 37,881 answer rows, 36,446 unique questions |
| 03 | Appendix B | DAPO diagnosis | 9,501+100 RL subset; Qwen2.5 also 35,974+100 |
| 04 | Appendix D | 7-category matched LoRA-SFT ablation | TACO + LeetCode-style |
| 05 | Appendix D | 7-category matched DAPO ablation | execution-verified TACO + LeetCode-style |
| 06 | Appendix A | full-parameter SFT robustness probe | same data as Stage 04 |

`plan/` and `progress/` contain exactly one document for each stage. Plans
describe the frozen design; progress files describe what was actually run and
the provenance of the reported results.

## Important provenance correction

Appendix B is not one experiment on an identical 36,074-example split. The
archived artifacts establish three distinct inputs:

- Stage 02 SFT: 37,881 answer rows (36,446 unique questions).
- Stage 03 DAPO-9.6k: 9,501 train + 100 validation problems.
- Stage 03 Qwen2.5 full-corpus DAPO: 35,974 train + 100 validation problems.

The configurations and progress records in this repository preserve those
actual contracts instead of repeating the inaccurate unified-data wording.

## Layout

```text
analysis/      result consolidation and paper statistics
configs/       path-free, machine-readable experiment contracts
dapo/          shared VERL/DAPO runner and archived runtime compatibility patches
docs/          data, evaluation, provenance, and reproducibility notes
evaluation/    current and legacy evaluation harnesses (kept separate)
metadata/      frozen counts, schemas, and cryptographic artifact identities
plan/          six frozen experimental plans
progress/      six evidence-backed completion records
reference_results/ hash-frozen aggregate packages (no prompts/generations)
stages/        data preparation and training code for stages 01--06
tests/         repository and lightweight unit checks
```

## Setup

1. Install [Pixi](https://pixi.sh/) on a Linux host with CUDA 12.x.
2. Run `pixi install` for SFT/evaluation. DAPO additionally requires the
   archived-compatible VERL recipe described in `docs/REPRODUCIBILITY.md`.
3. Copy `.env.example` to `.env`, or export the six required `CG_*` variables.
4. Put datasets and models outside this repository.
5. Validate the checkout:

```bash
pixi run validate-repo
pixi run test
```

Recreate the main-paper tables, heatmaps, headline figure, and null analyses
from the included aggregate scores:

```bash
pixi run python analysis/reproduce_paper.py \
  --package-root reference_results/stage01 \
  --output /external/reproduced-paper-artifacts \
  --draws 2000 \
  --label-shuffle-draws 20000
```

The same checkout also rebuilds Appendix A, Appendix B, and legacy Appendix D
from `reference_results/stage06`, `reference_results/appendix_b`, and
`reference_results/legacy`; exact commands are in
`reference_results/README.md`.

For the full data-to-results chain, start with
`stages/stage01_main_kodcode_lora_ablation/README.md`. Its portable driver
validates the frozen data and evaluation manifests, constructs the exact 216
training / 234 evaluation job graph, and packages the results. Outputs and all
non-code assets are required to live outside the Git checkout.

Each stage README/plan gives explicit preparation, training, and evaluation
commands. Cluster submission wrappers are deliberately not tied to an ACP
account, tenant, job ID, or private mount.

## Evaluation warning

`evaluation/paper_suite.py` is the current 11-task harness used by the main
experiment. `evaluation/legacy_suite.py` reconstructs the older TACO/LeetCode
experiments. Stage 06 deliberately aggregates ten tasks (the shared suite
minus GSM8K); its recovered GSM8K n=250 and APPS outputs are diagnostics only.
Absolute scores across harness generations are not interchangeable.
Generated Python is executed by the code benchmarks; run evaluation inside an
appropriately isolated container.

## Release status

The code and experiment contracts are organized for public version control.
Before a public release, the authors still need to choose a source-code
license, publish access instructions for the non-redistributed datasets and
models, and fill the exact historical model Hub revisions that could not be
recovered from local snapshots. See `docs/REPRODUCIBILITY.md` for the complete
known-gap ledger.

For a table/figure-level index, see `docs/PAPER_TRACEABILITY.md`; for the final
paper-to-`/data`-to-AOSS file audit, see `docs/SOURCE_AUDIT.md`.
