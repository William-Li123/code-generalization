# Stage 01 — main KodCode LoRA-SFT ablation

This is the main six-backbone, ten-category matched-control experiment. The
frozen design is in `../../plan/stage01_main_kodcode_lora_ablation.md`, the
machine-readable settings are in
`../../configs/stage01_main_kodcode_lora_ablation.json`, and the exact corpus,
arm, and benchmark hashes are under `../../metadata/`.

All data, models, checkpoints, generations, and result packages must live
outside this Git checkout.

## 1. Rebuild the reviewed v2 corpus

The label-only form audits taxonomy mapping but does **not** create the
materialized Parquet consumed by `build_arms.py`. A complete rebuild supplies
the reviewed labels and all four materialized source tables:

```bash
export CG_DATA_ROOT=/external/code-generalization-data

python stages/stage01_main_kodcode_lora_ablation/build_v2.py \
  --labels "$CG_DATA_ROOT/classification_v1/state/labels.parquet" \
  --train-sft "$CG_DATA_ROOT/classification_v1/output/train/sft_labeled.parquet" \
  --train-rl "$CG_DATA_ROOT/classification_v1/output/train/rl_labeled.parquet" \
  --validation-sft "$CG_DATA_ROOT/classification_v1/output/validation/sft_labeled.parquet" \
  --validation-rl "$CG_DATA_ROOT/classification_v1/output/validation/rl_labeled.parquet" \
  --taxonomy stages/stage01_main_kodcode_lora_ablation/taxonomy.json \
  --output-dir "$CG_DATA_ROOT/kodcode_clean_36074" \
  --expected-rows 36074
```

The historical v1 classifier/API-labelled records are external and are not
redistributed. If the complete export is available, `--v1-root` replaces the
five explicit v1 paths above.

## 2. Build and verify the twelve arms

```bash
python stages/stage01_main_kodcode_lora_ablation/build_arms.py \
  --source "$CG_DATA_ROOT/kodcode_clean_36074/train/sft_labeled.parquet" \
  --output-dir "$CG_DATA_ROOT/stage01_arms" \
  --seed 20260730
```

The generated `manifest.json` must match
`../../metadata/stage01/arm_manifest.json`. `run_stage01.py` checks all counts,
problem-ID set hashes, manifest hashes, and—unless explicitly skipped for a
fast dry run—every Parquet SHA256.

## 3. Prepare the exact evaluation snapshot

```bash
export CG_DATA_TEST_ROOT="$CG_DATA_ROOT/paper_standard11"

python evaluation/prepare_standard11.py \
  --source-root /external/preserved-normalized-benchmarks \
  --output-dir "$CG_DATA_TEST_ROOT" \
  --mbppplus-revision IMMUTABLE_HUB_COMMIT \
  --mbpp-revision IMMUTABLE_HUB_COMMIT \
  --scienceqa-revision IMMUTABLE_HUB_COMMIT
```

ARC-C, FinQA, HumanEval, LegalBench, MedCalc, and the two MATH slices require
the preserved normalized files because their exact historical normalization
revisions were not recovered. The script refuses all row/hash mismatches.

## 4. Configure model snapshots

Copy `../../configs/stage01_model_paths.example.json` outside the repository,
replace each value with a local model snapshot, and record immutable Hub
revisions in `../../configs/model_registry.json` before a new public rerun. The
historical commit revisions were not retained and are deliberately not guessed.

The following reusable arguments are used by every driver action:

```bash
COMMON_ARGS="\
--model-paths /external/contracts/stage01_model_paths.json \
--arms-root $CG_DATA_ROOT/stage01_arms \
--eval-data-root $CG_DATA_TEST_ROOT \
--checkpoint-root /external/outputs/stage01/checkpoints \
--eval-root /external/outputs/stage01/evaluation \
--summary-root /external/outputs/stage01/summaries \
--release-root /external/outputs/stage01/release \
--paper-output-root /external/outputs/stage01/paper \
--matrix /external/outputs/stage01/job_matrix.json"
```

## 5. Validate templates and build the formal job matrix

```bash
python stages/stage01_main_kodcode_lora_ablation/run_stage01.py \
  validate-inputs $COMMON_ARGS

python stages/stage01_main_kodcode_lora_ablation/run_stage01.py \
  preflight $COMMON_ARGS \
  --preflight-output /external/outputs/stage01/template_preflight.json

python stages/stage01_main_kodcode_lora_ablation/run_stage01.py \
  build-matrix $COMMON_ARGS
```

The matrix contains exactly 216 one-GPU training jobs and 234 one-GPU
evaluation jobs. The historical four-GPU ACP job was four independent arm
lanes, not one four-process DDP training. Each training process therefore uses
per-device batch 1 and gradient accumulation 16, for effective batch 16.

Run a matrix entry locally or from a scheduler array:

```bash
python stages/stage01_main_kodcode_lora_ablation/run_stage01.py \
  run-job $COMMON_ARGS --job-index "$JOB_INDEX"
```

Respect each evaluation job's `depends_on` field. Base evaluations have no
training dependency. Formal evaluation uses native chat templates, Qwen
thinking disabled, temperature 0.2, top-p 0.95, and the strict 11-task
manifest. The evaluator refuses a mismatched resume file or benchmark hash.

## 6. Summarize and reproduce paper artifacts

After all jobs have completed:

```bash
python stages/stage01_main_kodcode_lora_ablation/run_stage01.py \
  postprocess $COMMON_ARGS \
  --draws 2000 \
  --label-shuffle-draws 20000
```

This creates the 18 model/seed summaries, three 78-row seed CSVs, the 78-row
average CSV, `package_manifest.json`, support tables, additive residuals,
raw/residual permutation nulls, task-sensitivity label shuffle, leave-one-
backbone-out checks, and all reconstructed manuscript-named figure/table
outputs.
