# Stage 01 plan — main KodCode LoRA-SFT ablation

## Objective

Estimate category support by comparing a 30,353-row random-drop control with
ten equally sized leave-one-primary-category-out arms across six backbones.

## Frozen design

- Data: 35,974 clean train rows and 100 validation rows.
- Arms: full, matched random control, and ten leave-one-out arms.
- Models: the six models listed in the matching config.
- Seeds: 20260730, 20260731, 20260801.
- LoRA-SFT: one epoch, LR 2e-5, effective batch 16, rank/alpha/dropout
  16/32/0.05, native template, assistant-only loss, no truncation beyond 4096.
- Evaluation: current 11-task suite; three evaluation seeds; pass@1.

## Procedure

1. Validate labelled Parquet schema and category counts.
2. Build deterministic matched arms with `build_arms.py`.
3. Run template preflight for every model.
4. Train all 12 arms for each model and seed with `train_lora.py`.
5. Evaluate Base and every formal adapter with `evaluation/paper_suite.py`.
6. Consolidate seed CSVs and compute matched-control support statistics.

## Expected artifacts

External outputs contain arm manifests, one final adapter per training run,
per-task metrics, per-seed summaries, and the three-seed aggregate. No artifact
is committed to Git.

