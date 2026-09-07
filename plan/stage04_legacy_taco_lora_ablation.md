# Stage 04 plan — legacy TACO LoRA-SFT ablation

## Objective

Reconstruct the legacy seven-category LoRA-SFT study on two backbones while
preserving the historical training contract and the current 11-task paper
evaluation standard. Data, models, adapters, generated responses, and metrics
remain outside the repository.

## Frozen design

- Full pool: 21,879 rows; matched arms: 14,043 rows.
- Arms: `full_sft`, `full_sft_14043`, and seven leave-category-out arms.
- Qwen2.5-7B-Instruct and Qwen3-8B-Base.
- Seeds 20260603, 20260604, 20260605.
- Formal matrix: 2 models × 9 arms × 3 seeds = 54 LoRA training jobs.
- One-epoch LoRA-SFT, LR 1e-4, assistant-only loss.
- `full_sft` is the historical full-data LoRA name, not full-parameter SFT.

The seven removal arms are `no_math_number_theory`, `no_data_structure`,
`no_dynamic_programming`, `no_greedy_search`,
`no_implementation_simulation`, `no_graph`, and `no_other_algorithm`.

## Reproducibility contract

- `CG_DATA_ROOT`, `CG_DATA_TEST_ROOT`, `CG_MODEL_ROOT`, and `CG_OUTPUT_ROOT`
  point to external artifacts. `CG_WORK_ROOT` is optional scratch space.
- The committed Stage-04 metadata is compact: counts, record schema, source
  provenance, and SHA-256 placeholders only. It must never contain sample IDs,
  JSONL rows, model weights, or result payloads.
- `build_job_matrix.py` is the canonical portable expansion of the 54 training
  combinations. The multi-seed shell runner uses the same default dimensions
  and constructs its completion summary from submitted manifest rows.

## Procedure

Build the TACO/LeetCode-style pool and category index, generate deterministic
arm JSONL files, verify the compact counts/hashes, expand the training matrix,
and train all formal adapters. Then build the evaluation matrix and consume it
through `evaluation/legacy_suite.py`.

Formal evaluation uses `--tasks scoreable --limit 0`, the common 11-dataset
suite. Evaluate each of the two base models exactly once, then evaluate all 54
adapters. Thus the default evaluation matrix has 56 jobs, not 60. Reuse each
base result across the three training seeds.

For each task and seed, compute category support as:

```text
U(category, task) = score(full_sft_14043, task) - score(no_category, task)
```

Report arm scores and paired support values as mean ± standard deviation across
the three seeds. APPS-Hard, PlanBench, HealthBench, and HealthBench Professional
are supplementary/diagnostic only and must not enter the formal overall. Do not
mix Stage-04 legacy absolute scores with Stage 01.
