# Stage 04 — legacy TACO LoRA-SFT ablation

This directory reconstructs the Appendix-D legacy TACO + LeetCode-style
LoRA-SFT experiment without bundling datasets, model weights, adapters, logs, or
per-example identifiers. The formal training matrix is fixed by default at:

```text
2 models x 9 arms x 3 seeds = 54 training jobs
models: Qwen2.5-7B-Instruct, Qwen3-8B-Base
seeds:  20260603, 20260604, 20260605
arms:   full_sft, full_sft_14043, and seven no_<category> arms
```

`full_sft` is a historical name for the 21,879-row full-data **LoRA** arm; it
does not mean full-parameter tuning. Every other arm has 14,043 rows. Category
support must compare `full_sft_14043` with the corresponding `no_*` arm.

## External roots

Set all large artifacts outside this repository:

```bash
export CG_DATA_ROOT=/path/to/training-data
export CG_DATA_TEST_ROOT=/path/to/data_test
export CG_MODEL_ROOT=/path/to/base-models
export CG_OUTPUT_ROOT=/path/to/outputs
export CG_WORK_ROOT=/path/to/scratch  # optional; defaults to $CG_OUTPUT_ROOT/.work
```

The evaluation harness executes model-generated Python. Run it only inside an
appropriately isolated environment; generated code must be treated as
untrusted.

## 1. Rebuild the arm files

The source datasets must already exist below `CG_DATA_ROOT`; the repository does
not download or redistribute them.

```bash
python stages/stage04_legacy_taco_lora_ablation/prepare_taco_code_data.py \
  --taco-dir "$CG_DATA_ROOT/raw_data/TACO" \
  --output-dir "$CG_DATA_ROOT/code_data" \
  --overwrite
python stages/stage04_legacy_taco_lora_ablation/prepare_leetcode_taco_fusion.py \
  --leetcode-dir "$CG_DATA_ROOT/raw_data/newfacade_LeetCodeDataset_34803eb64eab" \
  --code-data-dir "$CG_DATA_ROOT/code_data" \
  --overwrite
python stages/stage04_legacy_taco_lora_ablation/prepare_stage2_taco_sft_data.py \
  --code-data-dir "$CG_DATA_ROOT/code_data/merged_taco_leetcode_by_primary" \
  --output-dir "$CG_DATA_ROOT/stage2_taco_sft_clean_prompt" \
  --overwrite
```

If the external source checkouts use different directory names, change only
the two raw input paths; keep all generated data below `CG_DATA_ROOT`.

The final files are expected at
`$CG_DATA_ROOT/stage2_taco_sft_clean_prompt/<arm>.jsonl`. The committed compact
manifest in `../../metadata/stage04/` records counts, schema, provenance, and
hash placeholders only. Populate its hashes after rebuilding; do not add IDs or
data rows to Git.

## 2. Inspect and run the formal training matrix

`build_job_matrix.py` is account-neutral and emits paths derived only from the
external roots:

```bash
python stages/stage04_legacy_taco_lora_ablation/build_job_matrix.py \
  --data-root "$CG_DATA_ROOT" \
  --model-root "$CG_MODEL_ROOT" \
  --output-root "$CG_OUTPUT_ROOT" \
  --work-root "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}" \
  --output "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage04_training_matrix.json"
```

The default manifest must report 54 jobs. The existing multi-seed driver
consumes the same frozen dimensions and writes adapters under explicit seed
directories:

```bash
bash stages/stage04_legacy_taco_lora_ablation/run_stage2_lora_sft_multi_seed_acp.sh
```

Use `SEEDS`, `MODELS`, or comma-separated `ONLY_ARMS` only for deliberate
partial/resume runs. With no overrides, the driver runs all seeds
`20260603 20260604 20260605` and its final summary is derived dynamically from
the submitted rows rather than from hard-coded dimensions.

## 3. Build and consume the formal evaluation matrix

Only `../../evaluation/legacy_suite.py --tasks scoreable --limit 0` is formal.
The `scoreable` alias is the canonical 11-dataset paper suite; APPS-Hard,
PlanBench, and HealthBench are not part of this overall.

```bash
python stages/stage04_legacy_taco_lora_ablation/build_evaluation_matrix.py \
  --data-test-root "$CG_DATA_TEST_ROOT" \
  --model-root "$CG_MODEL_ROOT" \
  --output-root "$CG_OUTPUT_ROOT" \
  --output "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage04_evaluation_matrix.json"

# Inspect one command without loading a model.
python stages/stage04_legacy_taco_lora_ablation/run_evaluation_matrix.py \
  --matrix "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage04_evaluation_matrix.json" \
  --job-index 0 --dry-run

# Run one scheduler-selected row, or use --all for a sequential local run.
python stages/stage04_legacy_taco_lora_ablation/run_evaluation_matrix.py \
  --matrix "${CG_WORK_ROOT:-$CG_OUTPUT_ROOT/.work}/stage04_evaluation_matrix.json" \
  --job-index "${JOB_INDEX}" --resume
```

The default evaluation matrix contains 56 rows: 54 adapters plus exactly one
base evaluation per model. Base rows use evaluation seed `20260603` and are
reused across all three adapter training seeds; they must not be expanded to
six base rows during aggregation.

## Outputs and aggregation contract

Adapters are expected under:

```text
$CG_OUTPUT_ROOT/checkpoints/sft_lora/stage2_taco_sft_clean_prompt/
  seed_<seed>/<model>/<arm>/
```

Metrics are written under:

```text
$CG_OUTPUT_ROOT/evaluation/stage04_legacy_taco_lora_ablation/
  base/<model>/metrics.json
  seed_<seed>/<model>/<arm>/metrics.json
```

Report each adapter arm as mean ± standard deviation across the three training
seeds. Compute category support per seed as
`score(full_sft_14043) - score(no_category)`, then aggregate those paired
differences. Do not mix these legacy absolute scores with Stage 01.
