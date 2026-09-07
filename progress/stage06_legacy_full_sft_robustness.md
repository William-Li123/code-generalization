# Stage 06 progress

**Status:** completed as a robustness probe and used in Appendix A.

- Two backbones and the legacy Stage-04 data contract were used.
- The recovered checkpoint/evaluation chain identifies training seeds
  20260603/20260604 and evaluation seed 20260605. The paper instead reports
  training seeds 20260604/20260605; this discrepancy is now explicit.
- Full-parameter SFT caused severe catastrophic forgetting, particularly on
  HumanEval and MBPP, and was therefore not used for the main support-map
  analysis.
- Intermediate group/shard, GSM8K n=250, and APPS files exist; they are legacy
  diagnostics. Paper Figure 7 averages the other ten canonical tasks, and the
  restored packager enforces that exact contract.

Evidence: `checkpoint/sft_full/stage4_taco_full_sft/` and AOSS evaluation
component `03962`.
