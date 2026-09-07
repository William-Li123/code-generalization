# Reference result package

This directory contains only the small, aggregate per-seed score tables needed
to reproduce the paper's analyses without repeating 216 GPU trainings and 234
evaluations. It contains no prompts, benchmark records, generations, hidden
tests, model weights, adapters, or checkpoints.

The Stage-01 CSVs are normalized to UTF-8 without BOM and LF line endings. Each
seed file has exactly 78 rows (`6 models × 13 evaluated conditions`); the
average file has the matching 78 cells averaged over the three formal seeds.
Their hashes and schema are frozen in `stage01/package_manifest.json`.

`appendix_b/` contains the 27 selected aggregate metric/selection JSON files
needed for Tables 3/4 and Figures 8/9. Historical machine paths are replaced by
an external-root placeholder; all scores, prompt contracts, row counts and
selection values remain unchanged. Rebuild the Appendix-B artifacts with:

```bash
python stages/stage03_kodcode_dapo_diagnosis/make_appendix_b_outputs.py \
  --stage02-work reference_results/appendix_b/source/stage02_work \
  --rl10k-work reference_results/appendix_b/source/rl10k_work \
  --full-work reference_results/appendix_b/source/full_work \
  --output-dir /external/reproduced-appendix-b
```

`legacy/` contains the canonical-GSM8K aggregate tables used by Appendix D,
plus the 94-file GSM8K replacement audit. Rebuild the legacy figures and
statistics without raw generations with:

```bash
python analysis/package_legacy_results.py \
  --reference-package reference_results/legacy \
  --output /external/reproduced-legacy-appendix
```

`stage06/` is reconstructed directly from archive component 03962 and contains
the two recovered full-SFT seed tables plus their average under Figure 7's
ten-task no-GSM8K contract. Rebuild Appendix A with:

```bash
python stages/stage06_legacy_full_sft_robustness/package_appendix_a.py \
  --reference-package reference_results/stage06 \
  --output-dir /external/reproduced-appendix-a
```

Rebuild all main-paper artifacts directly from the checkout:

```bash
python analysis/reproduce_paper.py \
  --package-root reference_results/stage01 \
  --output /external/reproduced-paper-artifacts \
  --draws 2000 \
  --label-shuffle-draws 20000
```

The original manuscript-named analysis scripts were absent from both recovered
archives. Their public replacements are transparent reconstructions from the
published equations; see `docs/REPRODUCIBILITY.md` for the remaining per-cell
residual-null discrepancy.
