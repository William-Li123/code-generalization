# Reproducibility guide

## 1. Configure external storage

Export the variables in `.env.example`. No stage should assume a private
`/data/yuzheng/...` or `/mnt/aoss...` path.

## 2. Freeze manifests before training

Every data-building script writes or consumes a manifest containing row counts,
seeds, category counts, and hashes. Archive that manifest with results. Do not
rebuild an arm in place after a training job starts.

## 3. Keep prompt contracts matched

- Current chat/instruct SFT uses each tokenizer's native chat template.
- Qwen thinking is disabled for training and evaluation.
- Stage 02 deliberately compares `plain` and `native` contracts.
- Qwen3-8B-Base in Stage 03 uses the plain contract.
- Only assistant answer tokens receive SFT loss.

Run the relevant template/token-length preflight before scheduling GPUs.

## 4. Training and evaluation are separate

Training creates adapters or full checkpoints; evaluation consumes immutable
formal checkpoints. Test scores must not select Stage-01 SFT checkpoints.
Stage-03 DAPO adapter selection follows its frozen validation rule and must be
recorded before the 11-task test run.

## 5. DAPO environment

The archived DAPO runs used VERL 0.8.0 and external recipe commit
`e0f4dc2ccdd51e8e445a683b828090577c625534`, plus the compatibility patches in
`dapo/runtime_patches`. Apply them to an isolated environment, never to a
shared system installation. `dapo/run_training.sh` exposes all formal settings
through environment variables. The VERL recipe itself is an external
dependency and is not vendored here; see `dapo/PATCHES.md`.

## 6. Result aggregation and paper artifacts

Stage 01 has 12 trained arms and one Base condition. It therefore contains 216
LoRA trainings (`6 models x 12 arms x 3 seeds`) and 234 evaluation rows when
Base is evaluated under each seed (`6 x 13 x 3`). Do not call these 234
independent training runs.

Keep current and legacy evaluator outputs in distinct roots. The analysis
scripts reject missing arms and unexpected score ranges before plotting.

The small aggregate packages under `reference_results/` let a CPU-only
checkout rebuild the main analysis, Appendix-A/B artifacts, and legacy
Appendix-D figures without rerunning the GPU pipeline. Their manifests fix the
schema, row counts, byte sizes, and SHA256 hashes.

The manuscript names `make_paper_figures.py`, `make_overall_table.py`,
`make_support_table.py`, `make_headline_figure.py`, `make_extra_figures.py`, and
`make_residual_null.py`, but none was present in either recovered archive.
Their public replacements and `analysis/support_statistics.py` are transparent
reconstructions from the published equations, not byte-for-byte recoveries.
All entry points share `analysis/paper_artifacts.py`, fix their random seeds,
and emit the full per-seed, per-backbone, and per-cell tables needed to audit
the calculations.

Run the complete reconstruction with:

```bash
python analysis/reproduce_paper.py \
  --package-root reference_results/stage01 \
  --output /external/reproduced-paper-artifacts \
  --draws 2000 \
  --label-shuffle-draws 20000
```

The released seed tables reproduce the headline support magnitude and the
task-sensitivity result (mean pairwise correlation about 0.522), as well as the
absence of BH-FDR significant cells. The reconstructed residual-null algorithm
reproduces the global conclusion but not three manuscript Table-5 per-cell
empirical p-values exactly. With 20,000 draws the recovered cells are about
0.395, 0.279, and 0.572, versus 0.30, 0.31, and 0.26 in the manuscript. Because
the original implementation is absent, this discrepancy is recorded instead
of tuning the reconstruction to the printed values. The released score tables
also yield one additional uncorrected `p < 0.05`, bias-absorbed cell
(`graph_structures_and_stateful_systems` / FinQA, `p=0.049138`), so the
reconstruction emits nine Table-6-style rows while the manuscript prints eight.

## 7. What remains external or irrecoverable

The paper and recovered execution record contain four material provenance
differences, all preserved rather than silently normalized:

- Appendix-B prose calls the SFT source 36,074 problems; its archived input has
  37,881 answer rows and 36,446 unique original questions.
- Stage-03 prose reports LoRA dropout 0.05; the archived VERL invocation omitted
  it and PEFT therefore used 0.0.
- Stage-05 prose reports seed 20260610; the two complete-corpus references use
  archived seed 20260609, while the 16 matched runs use 20260610.
- Stage-06 prose reports train seeds 20260604/20260605; recovered train seeds
  are 20260603/20260604 and 20260605 is the evaluation seed. Figure 7 averages
  ten tasks and excludes both the GSM8K n=250 and APPS diagnostics.

Public reproduction additionally requires:

- the training records and frozen Parquet arms identified by the manifests in
  `metadata/`;
- the eleven benchmark JSONL files identified by
  `metadata/evaluation/paper_suite_reference.json`;
- local model snapshots matching `configs/model_registry.json` (the model IDs
  were recovered, but exact historical Hub commit revisions were not; use
  `configs/model_snapshot_manifest.py` to build or verify a complete
  content-addressed identity including every weight shard);
- the external VERL recipe at the commit recorded in `CODE_PROVENANCE.json`;
- the API-labelled legacy category data for exact Stage-04/05 reconstruction.

Raw datasets, hidden tests, model weights, adapters, generations, and grader
traces are deliberately not redistributed. The paper source also contains an
unfilled training-hyperparameter placeholder; this repository reports only
settings supported by archived commands/configuration and does not invent the
missing author value.

## 8. Reproduction levels

- **Analysis reproduction:** self-contained for the main analysis, Appendices
  A/B, and legacy Appendix D; use the included aggregate packages.
- **Stage-01 execution reproduction:** code-complete once the hash-matched
  external datasets and local model snapshots are supplied.
- **Stages 02--06 historical reproduction:** archived recipes and contracts are
  retained, but exact reruns also depend on the external legacy assets listed
  above.
