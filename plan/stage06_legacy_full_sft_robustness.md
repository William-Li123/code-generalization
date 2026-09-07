# Stage 06 plan — legacy full-parameter SFT robustness

## Objective

Test whether the legacy code-task degradation is specific to LoRA capacity or
persists under full-parameter SFT.

## Frozen design

- Reuse Stage-04 full and matched arm files.
- Two backbones. Recovered training seeds are 20260603/20260604; the manuscript
  reports 20260604/20260605, while 20260605 is the recovered evaluation seed.
  Both facts are retained as an explicit provenance discrepancy.
- Full-parameter BF16 SFT, one epoch, LR 2e-5, effective batch 8, assistant-only
  loss, 4096-token limit.
- Evaluate with `appendix_a_10_no_gsm8k`: the ten Figure-7 tasks are the shared
  canonical suite minus GSM8K. Recovered GSM8K n=250 and APPS outputs are
  archive diagnostics, not Figure-7 inputs.

## Procedure

Generate the strict 36-cell matrix, train via four-GPU FSDP into seed-scoped
paths, and save final BF16 Hugging Face checkpoints without retaining multiple
optimizer states. Actually load every exported checkpoint, evaluate all 36
cells plus two Base references, reject incomplete task/sample matrices, and
package paired absolute and delta-vs-Base Appendix-A heatmaps. Stage-04 LoRA
comparison remains a separately labelled cross-regime analysis.
