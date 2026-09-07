# Stage 02 progress

**Status:** completed in the legacy project and used in Appendix B.

- Actual input: 37,881 answer rows, not the later 36,074-row corpus.
- Four models were evaluated at 0.5 and 1.0 epoch.
- Archived `diagnose_sft/RESULTS.md` values exactly match the SFT columns in the
  paper's Appendix-B table after rounding.
- No evidence was found for a later four-model rerun on 35,974 clean train rows.
- The clean implementation now freezes both source hashes, makes native
  `enable_thinking=False` identical in training/audit/evaluation, and emits a
  strict paper-row manifest instead of treating all eight prompt sweeps as
  paper conditions.

Evidence: AOSS component `03664`.
