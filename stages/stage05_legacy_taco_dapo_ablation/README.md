# Stage 05 — legacy TACO DAPO ablation

This stage has two provenance-distinct parts. The two complete-corpus
`full_dapo` references use the recovered seed `20260609`; the formal matched
attribution matrix contains `2 models × 8 arms = 16` jobs at seed `20260610`.
They must not be described as one 18-job, single-seed experiment.

All datasets, models, training outputs, and work files stay outside this
checkout. Set `CG_DATA_ROOT`, `CG_MODEL_ROOT`, `CG_OUTPUT_ROOT`,
`CG_DATA_TEST_ROOT`, and `CG_VERL_RECIPE_DIR`; `CG_WORK_ROOT` may be omitted and
then defaults to `${CG_OUTPUT_ROOT}/.work`.

## Prepare and freeze data

The verified full pool is finalized before the matched arms are sampled:

```bash
python stages/stage05_legacy_taco_dapo_ablation/prepare_stage3_dapo_full_data.py \
  --data-path "$CG_DATA_ROOT/clean_data/merged_clean_rl_dedup.jsonl" \
  --model-paths "$CG_MODEL_ROOT/Qwen3-8B-Base" "$CG_MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --output-dir "$CG_DATA_ROOT/stage3_dapo_full_verified"
python stages/stage05_legacy_taco_dapo_ablation/finalize_stage3_dapo_full_data.py \
  --data-dir "$CG_DATA_ROOT/stage3_dapo_full_verified" \
  --model-paths "$CG_MODEL_ROOT/Qwen3-8B-Base" "$CG_MODEL_ROOT/Qwen2.5-7B-Instruct"
python stages/stage05_legacy_taco_dapo_ablation/prepare_stage3_dapo_ablation_data.py \
  --full-data-dir "$CG_DATA_ROOT/stage3_dapo_full_verified" \
  --index-path "$CG_DATA_ROOT/code_data/merged_taco_leetcode_train_index.jsonl" \
  --output-root "$CG_DATA_ROOT/stage3_dapo_ablation_12168" \
  --overwrite
```

Archive the generated external manifests and file hashes with the run. The
compact public contract is `metadata/stage05/experiment_manifest.json`; no
Parquet, validation records, or selected problem IDs belong in Git.

## Train the formal matrix

Build the portable manifest. Its default contains two complete references at
seed `20260609` plus sixteen matched jobs at seed `20260610`:

```bash
python stages/stage05_legacy_taco_dapo_ablation/build_job_matrix.py \
  --data-root "$CG_DATA_ROOT" \
  --model-root "$CG_MODEL_ROOT" \
  --output-root "$CG_OUTPUT_ROOT" \
  --work-root "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}" \
  --output "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage05_jobs.json"
```

Inspect without launching, launch one job, or launch the full sequential
matrix:

```bash
python stages/stage05_legacy_taco_dapo_ablation/consume_job_matrix.py \
  --manifest "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage05_jobs.json" \
  --all --dry-run

python stages/stage05_legacy_taco_dapo_ablation/consume_job_matrix.py \
  --manifest "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage05_jobs.json" \
  --index 0

python stages/stage05_legacy_taco_dapo_ablation/consume_job_matrix.py \
  --manifest "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage05_jobs.json" \
  --all
```

`run_job.sh` verifies the actual Parquet row count and derives one-epoch steps
(`540` complete, `381` matched with batch 32). Qwen3 uses response length 3072;
Qwen2.5 uses 2048. It delegates runtime compatibility exclusively to
`../../dapo/run_training.sh`, which applies or checks the five recovered modules
inside the isolated `CG_VERL_ENV`, verifies their hashes, validates the pinned
recipe revision, and records the patch manifests with the external run. The
compatibility contract is documented in `../../dapo/PATCHES.md`; Stage 05 does
not maintain a second patch installer.

## Select and evaluate

`select_best_adapter.py` is strict: it considers only saved nonzero steps with
validation JSONL, maximizes mean `pass_frac`, and never falls back to test
scores or an unvalidated latest adapter. The matrix evaluator applies that rule
automatically and runs the canonical 11-task suite:

```bash
python stages/stage05_legacy_taco_dapo_ablation/evaluate_job_matrix.py \
  --training-manifest "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage05_jobs.json" \
  --output-root "$CG_OUTPUT_ROOT" \
  --model-root "$CG_MODEL_ROOT" \
  --data-test-root "$CG_DATA_TEST_ROOT"

python stages/stage05_legacy_taco_dapo_ablation/evaluate_job_matrix.py \
  --training-manifest "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage05_jobs.json" \
  --output-root "$CG_OUTPUT_ROOT" \
  --model-root "$CG_MODEL_ROOT" \
  --data-test-root "$CG_DATA_TEST_ROOT" \
  --all

python stages/stage05_legacy_taco_dapo_ablation/evaluate_job_matrix.py \
  --training-manifest "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage05_jobs.json" \
  --output-root "$CG_OUTPUT_ROOT" \
  --model-root "$CG_MODEL_ROOT" \
  --data-test-root "$CG_DATA_TEST_ROOT" \
  --validate-only
```

The first command only writes the 20-cell evaluation manifest: two Base cells,
two complete references, and sixteen matched cells. `--all` performs the GPU
evaluations; `--validate-only` rejects missing tasks, wrong sample counts, and
the obsolete 250-row GSM8K subset.
