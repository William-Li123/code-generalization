# External dataset contracts

No dataset is committed to this repository. The paths below are logical
contracts under `CG_DATA_ROOT`; deployments may map them to local disk, object
storage, or a read-only dataset mount.

## Current KodCode family

### Clean main corpus

Expected files:

```text
kodcode_clean_36074/
  train/sft_labeled.parquet
  validation/sft_labeled.parquet
  taxonomy.json
```

Required SFT columns are `problem_id`, `prompt`, `response`, and
`primary_category`. The frozen counts are 35,974 train and 100 validation
rows. Each train row has one primary category; auxiliary technique labels are
not used to construct the main paper arms.

`metadata/stage01/corpus_manifest.json` records the exact source identity.
`metadata/stage01/arm_manifest.json` records every one of the twelve trained
arms, including row/category counts, problem-ID set hashes, Parquet hashes, and
sampling seeds. A Stage-01 run refuses arms that differ from this reference.

### Appendix-B SFT diagnosis corpus

Expected logical files are `kodcode4o_r1_clean38k/train.jsonl` and
`kodcode4o_r1_clean38k/train.parquet`. The Stage-02 preparation code requires
both and checks their frozen SHA256 values; neither is a disposable duplicate.

The archived experiment retained 37,881 answer rows representing 36,446
unique original questions. It must not be described as the later 36,074-row
deduplicated corpus.

### Appendix-B DAPO corpora

Expected logical layouts:

```text
kodcode_dapo_rl10k/{train,val}.parquet       # 9,501 + 100
kodcode_dapo_full/{train,val}.parquet        # 35,974 + 100
```

The full corpus is derived from the 37,881-row source by original-question
deduplication, overlap removal, static checks, and exact-reward preflight.
The RL10K and full-corpus preparation drivers iterate exact-reward failures,
rebuild, run strict count/hash checks, and only then write `.data_ready`.
Unrecovered historical ID-set hashes remain explicit `null` values in the
Stage-03 config rather than being inferred from a newer rebuild.

## Legacy TACO + LeetCode-style family

Stage 04 builds a 21,879-row SFT pool (19,493 TACO and 2,386 LeetCode-style
rows) and 14,043-row matched arms. Stage 05 uses an independently
execution-verified pool of 17,249 train problems plus 100 validation problems,
with 12,168 rows in each matched arm.

The legacy seven-category labels are not equivalent to the main ten-category
taxonomy. Keep both schemas and their arm manifests separate.

## Exact evaluation snapshot

Run `evaluation/prepare_standard11.py` to assemble the external paper-suite
layout. It can publicly normalize GSM8K, MBPP+, MBPP-Simple, and ScienceQA;
the other seven tasks require the preserved normalized JSONL files through
`--source-root` because their exact historical normalization code or revision
was not recovered. All eleven outputs are checked against the archived row
counts and SHA256 values by default, preventing a convenient but non-equivalent
benchmark revision from being substituted silently.

## Data that must never enter Git

- raw or transformed training/evaluation records;
- prompts paired with hidden tests;
- API-labelled full corpora;
- model generations and grader traces;
- Parquet, JSONL, archives, and dataset caches.

Only schemas, counts, hashes, and synthetic fixtures belong in this repository.
The sole CSV exception is `reference_results/stage01/csv`: these are aggregate
benchmark scores with no prompts, tests, or generations, and their identities
are fixed by an adjacent package manifest.
