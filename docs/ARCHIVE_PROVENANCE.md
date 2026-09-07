# Archive provenance

This file maps public-stage names to the original active and AOSS evidence.
Paths are provenance only and are not runtime defaults.

## Stage 01

Active source:

```text
/data/yuzheng/code_generalization/
```

Read-only snapshot:

```text
/mnt/aoss-250010161/cloudplatform/models/code_generalization/archive_20260813_full/
```

Relevant namespaces are `data_train/kodcode_clean_36074_v2`,
`data_train/stage1_sft_code_ablation`, `checkpoint/sft_lora/stage1_code_ablation`,
and `output/stage1_sft_code_ablation_eval`.

## Stages 02--06

Canonical legacy archive:

```text
/mnt/aoss-250010161/cloudplatform/models/codegeneralization3/full_archive_20260729/
```

The archive contains `ARCHIVE_COMPLETE` and `VALIDATION_COMPLETE`; its
`index.jsonl` maps original relative paths to components.

| Stage | Components / paths |
|---|---|
| 02 | component `03664`, `diagnose_sft/` |
| 03 RL10K | component `03661`, `diagnose_dapo/` |
| 03 full | component `03662`, `diagnose_dapo_full37881/` |
| 04 data | component `03644`, `data_train/stage2_taco_sft_clean_prompt/` |
| 04 checkpoints | components `03624--03629`, `checkpoint/sft_lora/stage2_taco_sft_clean_prompt/` |
| 04 evaluation | components `03769--03771` |
| 05 data | component `03649`, `data_train/stage3_dapo_ablation_12168/` |
| 05 checkpoints | components `03611--03619` |
| 05 evaluation | component `03890` plus fixed-baseline follow-ups |
| 06 checkpoints | `checkpoint/sft_full/stage4_taco_full_sft/` |
| 06 evaluation | component `03962` |
| legacy source code | component `03975` |
| legacy plans/progress | components `03969` / `03970` |

### Legacy matrix and seed clarifications

- Stage 04 contains 54 LoRA trainings: two backbones, nine arms, and training
  seeds `20260603/20260604/20260605`. Base is a reference condition evaluated
  once per backbone, not an additional training run.
- Stage 05 is provenance-split. The two complete-corpus `full_dapo` references
  use recovered seed `20260609`; the 16 matched model/arm runs use seed
  `20260610`. The manuscript reports DAPO as a single seed-20260610 experiment,
  so public config and metadata retain both the paper statement and recovered
  execution record.
- Stage 06 contains 36 full-parameter checkpoints. The recovered training seeds
  are `20260603/20260604`; the recovered combined launcher uses `20260605` as
  the evaluation seed. The manuscript instead reports full-SFT training seeds
  `20260604/20260605`. This is a documented provenance discrepancy, not a value
  to resolve by silently renaming archived runs.
- Stage 06 raw summaries include a 250-row GSM8K diagnostic and APPS. They
  remain historical evidence only: Appendix-A Figure 7 uses
  `appendix_a_10_no_gsm8k`, the ten canonical tasks other than GSM8K. This
  differs intentionally from the 11-task contracts in Stages 01--05.

The compact files under `metadata/stage04`, `metadata/stage05`, and
`metadata/stage06` record counts, schemas, seed roles, component provenance,
and the external hashes that must be captured at run time. Unrecovered data or
checkpoint hashes are explicitly marked unresolved rather than guessed. Raw
records, per-problem IDs, model weights, optimizer shards, generations, logs,
private scheduler metadata, and environment archives remain excluded from the
public repository.

`archive_20260702_cleanup` contains early cleanup material and large debug or
resume checkpoints; it is not the canonical result archive. Directories named
`codeislogic_v1` and the earlier Magicoder/UltraChat GRPO experiments are
different projects and are not used by the paper.
