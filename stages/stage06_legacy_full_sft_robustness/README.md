# Stage 06 — legacy full-parameter SFT robustness

This is the Appendix-A catastrophic-forgetting probe, not the main training
method. It reuses the external Stage-04 arm files and trains full BF16 models
with four-GPU FSDP.

The recovered execution record differs from the manuscript: archived training
seeds are `20260603` and `20260604`; `20260605` is the legacy evaluation seed.
The manuscript reports training seeds `20260604/20260605`. Both claims are
preserved in config/metadata, but the executable archived matrix follows
`20260603/20260604` and contains `2 models × 9 arms × 2 seeds = 36` checkpoints.

Set `CG_DATA_ROOT`, `CG_MODEL_ROOT`, `CG_OUTPUT_ROOT`, and
`CG_DATA_TEST_ROOT`. `CG_WORK_ROOT` is optional and defaults to
`${CG_OUTPUT_ROOT}/.work`. Data, models, checkpoints, and results must remain
outside this repository.

## Build and train the matrix

Inspect the exact seed-scoped paths before allocating GPUs:

```bash
python stages/stage06_legacy_full_sft_robustness/build_job_matrix.py \
  --data-root "$CG_DATA_ROOT" \
  --model-root "$CG_MODEL_ROOT" \
  --output-root "$CG_OUTPUT_ROOT" \
  --work-root "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}" \
  --output "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage06_36_jobs.json"
```

Run both recovered training seeds. The 18-cell driver now defaults to a
seed-scoped experiment name, so the second seed cannot collide with the first:

```bash
bash stages/stage06_legacy_full_sft_robustness/run_two_seed_matrix.sh
```

By default, each completed 18-cell seed matrix is actually loaded with
`AutoModelForCausalLM.from_pretrained`. A cheaper file/run-manifest inspection
is available for preflight only:

```bash
python stages/stage06_legacy_full_sft_robustness/validate_full_checkpoints.py \
  --matrix "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage06_36_jobs.json"

python stages/stage06_legacy_full_sft_robustness/validate_full_checkpoints.py \
  --matrix "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage06_36_jobs.json" \
  --load \
  --output "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage06_load_validation.json"
```

Optimizer/resume states are external and may be discarded only after the
final Hugging Face export passes actual-load validation.

## Evaluate the formal matrix

The first command writes a 38-job evaluation manifest: 36 trained checkpoints
plus one Base evaluation per backbone. The second runs all jobs with the
Appendix-A ten-task contract and legacy evaluation seed `20260605`:

```bash
python stages/stage06_legacy_full_sft_robustness/evaluate_full_sft_matrix.py \
  --training-matrix "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage06_36_jobs.json" \
  --output-root "$CG_OUTPUT_ROOT" \
  --data-test-root "$CG_DATA_TEST_ROOT"

python stages/stage06_legacy_full_sft_robustness/evaluate_full_sft_matrix.py \
  --training-matrix "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage06_36_jobs.json" \
  --output-root "$CG_OUTPUT_ROOT" \
  --data-test-root "$CG_DATA_TEST_ROOT" \
  --all

python stages/stage06_legacy_full_sft_robustness/evaluate_full_sft_matrix.py \
  --training-matrix "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage06_36_jobs.json" \
  --output-root "$CG_OUTPUT_ROOT" \
  --data-test-root "$CG_DATA_TEST_ROOT" \
  --validate-only
```

Validation rejects missing tasks and wrong benchmark row counts. The contract
is the canonical paper suite with GSM8K removed: Figure 7 averages ten tasks.
Recovered GSM8K n=250 and APPS outputs are historical diagnostics only.

## Package Appendix A

Create strict per-seed and two-seed-mean CSVs plus paired absolute and
delta-vs-Base heatmaps:

```bash
python stages/stage06_legacy_full_sft_robustness/package_appendix_a.py \
  --eval-root "$CG_OUTPUT_ROOT/stage06_legacy_full_sft_robustness/evaluation/appendix_a_10_no_gsm8k" \
  --output-dir "$CG_OUTPUT_ROOT/stage06_legacy_full_sft_robustness/appendix_a_package"
```

For CPU-only artifact reproduction from the exact recovered component-03962
aggregate, use the included hash-frozen package:

```bash
python stages/stage06_legacy_full_sft_robustness/package_appendix_a.py \
  --reference-package reference_results/stage06 \
  --output-dir /external/reproduced-appendix-a
```

The package has the publication layout `csv/`, `picture/full_sft/`, and
`README.md`: two seed CSVs, one two-seed average CSV, and separate absolute and
delta-vs-Base PNGs for each model. Base is the first row in every model block,
and all displayed values are percentages.

`plot_full_sft_robustness.py` remains a generic diagnostic bar plot for
arbitrary JSON/CSV inputs. It is not the formal Appendix-A packager and does not
enforce the 36-cell matrix.
