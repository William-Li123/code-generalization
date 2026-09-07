# Stage 03 — KodCode DAPO diagnosis

This directory preserves the two distinct Appendix-B branches:

- `rl10k/`: four models, 9,501 train + 100 frozen validation problems;
- `full_corpus/`: Qwen2.5 only, 35,974 train + the same 100 validation problems.

Datasets, models, checkpoints, VERL, and outputs are external. The scripts
refuse a source hash/count mismatch, an incomplete exact-reward preflight, or a
missing `.data_ready` marker. Adapter selection is fail-closed unless an
operator explicitly requests the exploratory latest-adapter fallback.

## External roots

```bash
export CG_DATA_ROOT=/path/to/datasets
export CG_MODEL_ROOT=/path/to/models
export CG_OUTPUT_ROOT=/path/to/outputs
export CG_WORK_ROOT=/path/to/scratch
export CG_DATA_TEST_ROOT=/path/to/data_test
export CG_VERL_RECIPE_DIR=/path/to/pinned/verl-recipe-checkout
export CG_VERL_ENV=/path/to/isolated/verl-environment
# Or, instead of CG_VERL_ENV:
# export CG_VERL_ENV_ARCHIVE=/path/to/isolated-verl-environment.tar
```

The expected source is
`$CG_DATA_ROOT/kodcode4o_r1_clean38k/train.parquet` with SHA-256
`6e91ea43bb43ff9d20514c065428152078b36ba57c6f6c577f794395f6decae1`.
The archived VERL/DAPO recipe commit is
`e0f4dc2ccdd51e8e445a683b828090577c625534`.
`shared/run_job.sh` delegates to `dapo/run_training.sh`; that public runner is
the sole owner of applying and checking `dapo/apply_runtime_patches.py`.

## RL10K: prepare, verify, train, evaluate

The preparation driver repeatedly builds candidates, runs the exact binary
reward on every retained reference answer, accumulates all failure IDs, and
rebuilds until the final pass has zero failures. It writes `.data_ready` only
after the strict frozen count/hash and prompt/reward checks pass.

```bash
bash stages/stage03_kodcode_dapo_diagnosis/rl10k/prepare_rl10k_data.sh
bash stages/stage03_kodcode_dapo_diagnosis/rl10k/run_all_train_then_eval.sh
```

To resume a stopped preparation loop without discarding the cumulative reject
set, set `START_ROUND` to the next round. Formal adapter selection requires a
complete 100-row validation JSONL. Only exploratory work may opt into fallback:

```bash
python stages/stage03_kodcode_dapo_diagnosis/shared/select_best_adapter.py \
  --checkpoint-root /path/to/checkpoints \
  --output-root /path/to/output \
  --model-key qwen25 \
  --selection-file /path/to/exploratory-selection.json \
  --exploratory-fallback-latest
```

## Full corpus: fixed-validation override, prepare, train, evaluate

The full branch must reuse the exact RL10K validation parquet. The portable
default can be overridden explicitly with the freshly prepared RL10K output:

```bash
export FIXED_VALIDATION="$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/rl10k/data/val.parquet"
bash stages/stage03_kodcode_dapo_diagnosis/full_corpus/prepare_full_data.sh
bash stages/stage03_kodcode_dapo_diagnosis/full_corpus/run_train_then_eval.sh
```

`full_corpus/build_generic_dataset.py` is the recovered template-neutral
exporter used before the full-corpus branch: it joins verified SFT answers to
the RL split, checks zero train/validation overlap, writes paired SFT/RL
Parquets and freezes their counts and SHA256 hashes. It is kept in addition to
the later iterative `prepare_data.py` pipeline because both occur in the
archived experiment history.

Both preparation scripts accept `START_ROUND`, `MAX_ROUNDS`, preflight worker
and timeout overrides. A historical frozen ID-list hash can be supplied to each
`static_check.py` with `--expected-train-id-sha256` and
`--expected-validation-id-sha256`; file and ID hashes are always recorded in
the generated manifest and checked against the Parquet files. Set
`EXPECTED_FIXED_VALIDATION_SHA256` when the archived fixed-validation file hash
is available; the actual supplied file hash is always recorded.

## Appendix-B outputs

After Stage 02 and both Stage 03 evaluations are present, reconstruct Tables
3/4 and Figures 8/9 with strict prompt, task, selection, Base-reference, and
rounded paper-value checks. Independent DAPO Base reruns are hash-audited and
their decoding drift is recorded; the published deltas use the frozen Stage-02
Base row shared by Tables 3/4:

```bash
python stages/stage03_kodcode_dapo_diagnosis/make_appendix_b_outputs.py \
  --stage02-work "$CG_OUTPUT_ROOT/stage02_kodcode_sft_diagnosis" \
  --rl10k-work "$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/rl10k" \
  --full-work "$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/full_corpus" \
  --output-dir "$CG_OUTPUT_ROOT/stage03_kodcode_dapo_diagnosis/appendix_b"
```

Generated CSV, PNG, manifest, Parquet, checkpoints, models, and datasets stay
outside Git.
