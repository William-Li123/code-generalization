# Stage 03 progress

**Status:** completed with two different data branches and used in Appendix B.

- Four-model RL10K branch: 9,792 candidates, 9,601 accepted, 9,501 train + 100
  validation after static and exact-reward checks.
- Full branch: 35,974 train + 100 validation; only Qwen2.5 is confirmed
  complete, with final reported adapter at step 397.
- These DAPO inputs are not identical to Stage-02 SFT data.
- Both branches now expose a strict frozen source/count/generated-hash
  contract. RL10K has the same resumable cumulative-failure rebuild loop and
  `.data_ready` gate as the full branch.
- Formal adapter selection now fails closed when no complete 100-row
  validation candidate exists; latest-adapter fallback is explicitly marked
  exploratory.
- The compact historical manifest preserves the four recovered RL10K
  validation-selected steps and scores. The unrecovered full-corpus selected
  step remains null rather than silently equating "latest" with "best".

Evidence: AOSS components `03661` and `03662`.
